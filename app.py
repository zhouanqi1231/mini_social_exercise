from flask import Flask, render_template, request, redirect, url_for, session, flash, g
from werkzeug.security import generate_password_hash, check_password_hash
from cryptography.fernet import Fernet
import collections
import json
import sqlite3
import hashlib
import re
from datetime import datetime
import sqlite3
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# from nltk.corpus import stopwords
import pandas as pd
import sqlite3
from scipy.sparse import csr_matrix

app = Flask(__name__)
app.secret_key = "123456789"
DATABASE = "database.sqlite"

# Load censorship data
# WARNING! The censorship.dat file contains disturbing language when decrypted.
# If you want to test whether moderation works,
# you can trigger censorship using these words:
# tier1badword, tier2badword, tier3badword
ENCRYPTED_FILE_PATH = "censorship.dat"
fernet = Fernet("xpplx11wZUibz0E8tV8Z9mf-wwggzSrc21uQ17Qq2gg=")
with open(ENCRYPTED_FILE_PATH, "rb") as encrypted_file:
    encrypted_data = encrypted_file.read()
decrypted_data = fernet.decrypt(encrypted_data)
MODERATION_CONFIG = json.loads(decrypted_data)
TIER1_WORDS = MODERATION_CONFIG["categories"]["tier1_severe_violations"]["words"]
TIER2_PHRASES = MODERATION_CONFIG["categories"]["tier2_spam_scams"]["phrases"]
TIER3_WORDS = MODERATION_CONFIG["categories"]["tier3_mild_profanity"]["words"]


def get_db():
    """
    Connect to the application's configured database. The connection
    is unique for each request and will be reused if this is called
    again.
    """
    if "db" not in g:
        g.db = sqlite3.connect(DATABASE, detect_types=sqlite3.PARSE_DECLTYPES)
        g.db.row_factory = sqlite3.Row

    return g.db


@app.teardown_appcontext
def close_connection(exception):
    """Closes the database again at the end of the request."""
    db = g.pop("db", None)

    if db is not None:
        db.close()


def query_db(query, args=(), one=False, commit=False):
    """
    Queries the database and returns a list of dictionaries, a single
    dictionary, or None. Also handles write operations.
    """
    db = get_db()

    # Using 'with' on a connection object implicitly handles transactions.
    # The 'with' statement will automatically commit if successful,
    # or rollback if an exception occurs. This is safer.
    try:
        with db:
            cur = db.execute(query, args)

        # For SELECT statements, fetch the results after the transaction block
        if not commit:
            rv = cur.fetchall()
            return (rv[0] if rv else None) if one else rv

        # For write operations, we might want the cursor to get info like lastrowid
        return cur

    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return None


@app.template_filter("datetimeformat")
def datetimeformat(value):
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        dt = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    else:
        return "N/A"
    return dt.strftime("%b %d, %Y %H:%M")


REACTION_EMOJIS = {
    "like": "❤️",
    "love": "😍",
    "laugh": "😂",
    "wow": "😮",
    "sad": "😢",
    "angry": "😠",
}
REACTION_TYPES = list(REACTION_EMOJIS.keys())


@app.route("/")
def feed():
    #  1. Get Pagination and Filter Parameters
    try:
        page = int(request.args.get("page", 1))
    except ValueError:
        page = 1
    sort = request.args.get("sort", "new").lower()
    show = request.args.get("show", "all").lower()

    # Define how many posts to show per page
    POSTS_PER_PAGE = 10
    offset = (page - 1) * POSTS_PER_PAGE

    current_user_id = session.get("user_id")
    params = []

    #  2. Build the Query
    where_clause = ""
    if show == "following" and current_user_id:
        where_clause = "WHERE p.user_id IN (SELECT followed_id FROM follows WHERE follower_id = ?)"
        params.append(current_user_id)

    # Add the pagination parameters to the query arguments
    pagination_params = (POSTS_PER_PAGE, offset)

    if sort == "popular":
        query = f"""
            SELECT p.id, p.content, p.created_at, u.username, u.id as user_id,
                   IFNULL(r.total_reactions, 0) as total_reactions, is_repost, original_post
            FROM posts p
            JOIN users u ON p.user_id = u.id
            LEFT JOIN (
                SELECT post_id, COUNT(*) as total_reactions FROM reactions GROUP BY post_id
            ) r ON p.id = r.post_id
            {where_clause}
            ORDER BY total_reactions DESC, p.created_at DESC
            LIMIT ? OFFSET ?
        """
        final_params = params + list(pagination_params)
        posts = query_db(query, final_params)
    elif sort == "recommended":
        posts = recommend(current_user_id, show == "following" and current_user_id)
    else:  # Default sort is 'new'
        query = f"""
            SELECT p.id, p.content, p.created_at, u.username, u.id as user_id, is_repost, original_post
            FROM posts p
            JOIN users u ON p.user_id = u.id
            {where_clause}
            ORDER BY p.created_at DESC
            LIMIT ? OFFSET ?
        """
        final_params = params + list(pagination_params)
        posts = query_db(query, final_params)

    posts_data = []
    for post in posts:
        # Determine if the current user follows the poster
        followed_poster = False
        if current_user_id and post["user_id"] != current_user_id:
            follow_check = query_db("SELECT 1 FROM follows WHERE follower_id = ? AND followed_id = ?", (current_user_id, post["user_id"]), one=True)
            if follow_check:
                followed_poster = True

        # Determine if the current user reacted to this post and with what reaction
        user_reaction = None
        if current_user_id:
            reaction_check = query_db("SELECT reaction_type FROM reactions WHERE user_id = ? AND post_id = ?", (current_user_id, post["id"]), one=True)
            if reaction_check:
                user_reaction = reaction_check["reaction_type"]

        reactions = query_db("SELECT reaction_type, COUNT(*) as count FROM reactions WHERE post_id = ? GROUP BY reaction_type", (post["id"],))
        comments_raw = query_db("SELECT c.id, c.content, c.created_at, u.username, u.id as user_id FROM comments c JOIN users u ON c.user_id = u.id WHERE c.post_id = ? ORDER BY c.created_at ASC", (post["id"],))
        post_dict = dict(post)
        post_dict["content"], _ = moderate_content(post_dict["content"])
        comments_moderated = []
        for comment in comments_raw:
            comment_dict = dict(comment)
            comment_dict["content"], _ = moderate_content(comment_dict["content"])
            comments_moderated.append(comment_dict)

        # original post
        original_post = None
        if post["is_repost"] == True:
            original_post_id = post["original_post"]
            original_post = query_db(
                """
                SELECT p.id as id, p.content, p.created_at, u.username, u.id as user_id
                FROM posts p
                JOIN users u ON p.user_id = u.id
                WHERE p.id = ?
                """,
                (original_post_id,),
                one=True,
            )

        posts_data.append({"post": post_dict, "reactions": reactions, "user_reaction": user_reaction, "followed_poster": followed_poster, "comments": comments_moderated, "is_repost": post["is_repost"], "original_post": original_post})

    #  4. Render Template with Pagination Info
    return render_template("feed.html.j2", posts=posts_data, current_sort=sort, current_show=show, page=page, per_page=POSTS_PER_PAGE, reaction_emojis=REACTION_EMOJIS, reaction_types=REACTION_TYPES)  # Pass current page number  # Pass items per page


@app.route("/posts/new", methods=["POST"])
def add_post():
    """Handles creating a new post from the feed."""
    user_id = session.get("user_id")

    # Block access if user is not logged in
    if not user_id:
        flash("You must be logged in to create a post.", "danger")
        return redirect(url_for("login"))

    # Get content from the submitted form
    content = request.form.get("content")

    # Pass the user's content through the moderation function
    moderated_content = content

    # Basic validation to ensure post is not empty
    if moderated_content and moderated_content.strip():
        db = get_db()
        db.execute("INSERT INTO posts (user_id, content) VALUES (?, ?)", (user_id, moderated_content))
        db.commit()
        flash("Your post was successfully created!", "success")
    else:
        # This will catch empty posts or posts that were fully censored
        flash("Post cannot be empty or was fully censored.", "warning")

    # Redirect back to the main feed to see the new post
    return redirect(url_for("feed"))


@app.route("/posts/<int:post_id>/delete", methods=["POST"])
def delete_post(post_id):
    """Handles deleting a post."""
    user_id = session.get("user_id")

    # Block access if user is not logged in
    if not user_id:
        flash("You must be logged in to delete a post.", "danger")
        return redirect(url_for("login"))

    # Find the post in the database
    post = query_db("SELECT id, user_id FROM posts WHERE id = ?", (post_id,), one=True)

    # Check if the post exists and if the current user is the owner
    if not post:
        flash("Post not found.", "danger")
        return redirect(url_for("feed"))

    if post["user_id"] != user_id:
        # Security check: prevent users from deleting others' posts
        flash("You do not have permission to delete this post.", "danger")
        return redirect(url_for("feed"))

    # If all checks pass, proceed with deletion
    db = get_db()
    # To maintain database integrity, delete associated records first
    db.execute("DELETE FROM comments WHERE post_id = ?", (post_id,))
    db.execute("DELETE FROM reactions WHERE post_id = ?", (post_id,))
    # Finally, delete the post itself
    db.execute("DELETE FROM posts WHERE id = ?", (post_id,))
    db.commit()

    flash("Your post was successfully deleted.", "success")
    # Redirect back to the page the user came from, or the feed as a fallback
    return redirect(request.referrer or url_for("feed"))


@app.route("/u/<username>")
def user_profile(username):
    """Displays a user's profile page with moderated bio, posts, and latest comments."""

    user_raw = query_db("SELECT * FROM users WHERE username = ?", (username,), one=True)
    if not user_raw:
        abort(404)

    user = dict(user_raw)
    moderated_bio, _ = moderate_content(user.get("profile", ""))
    user["profile"] = moderated_bio

    posts_raw = query_db("SELECT id, content, user_id, created_at FROM posts WHERE user_id = ? ORDER BY created_at DESC", (user["id"],))
    posts = []
    for post_raw in posts_raw:
        post = dict(post_raw)
        moderated_post_content, _ = moderate_content(post["content"])
        post["content"] = moderated_post_content
        posts.append(post)

    comments_raw = query_db("SELECT id, content, user_id, post_id, created_at FROM comments WHERE user_id = ? ORDER BY created_at DESC LIMIT 100", (user["id"],))
    comments = []
    for comment_raw in comments_raw:
        comment = dict(comment_raw)
        moderated_comment_content, _ = moderate_content(comment["content"])
        comment["content"] = moderated_comment_content
        comments.append(comment)

    followers_count = query_db("SELECT COUNT(*) as cnt FROM follows WHERE followed_id = ?", (user["id"],), one=True)["cnt"]
    following_count = query_db("SELECT COUNT(*) as cnt FROM follows WHERE follower_id = ?", (user["id"],), one=True)["cnt"]

    #  NEW: CHECK FOLLOW STATUS
    is_currently_following = False  # Default to False
    current_user_id = session.get("user_id")

    # We only need to check if a user is logged in
    if current_user_id:
        follow_relation = query_db("SELECT 1 FROM follows WHERE follower_id = ? AND followed_id = ?", (current_user_id, user["id"]), one=True)
        if follow_relation:
            is_currently_following = True
    # --

    return render_template("user_profile.html.j2", user=user, posts=posts, comments=comments, followers_count=followers_count, following_count=following_count, is_following=is_currently_following)


@app.route("/u/<username>/followers")
def user_followers(username):
    user = query_db("SELECT * FROM users WHERE username = ?", (username,), one=True)
    if not user:
        abort(404)
    followers = query_db(
        """
        SELECT u.username
        FROM follows f
        JOIN users u ON f.follower_id = u.id
        WHERE f.followed_id = ?
    """,
        (user["id"],),
    )
    return render_template("user_list.html.j2", user=user, users=followers, title="Followers of")


@app.route("/u/<username>/following")
def user_following(username):
    user = query_db("SELECT * FROM users WHERE username = ?", (username,), one=True)
    if not user:
        abort(404)
    following = query_db(
        """
        SELECT u.username
        FROM follows f
        JOIN users u ON f.followed_id = u.id
        WHERE f.follower_id = ?
    """,
        (user["id"],),
    )
    return render_template("user_list.html.j2", user=user, users=following, title="Users followed by")


@app.route("/posts/<int:post_id>")
def post_detail(post_id):
    """Displays a single post and its comments, with content moderation applied."""

    post_raw = query_db(
        """
        SELECT p.id, p.content, p.created_at, u.username, u.id as user_id, is_repost, original_post
        FROM posts p
        JOIN users u ON p.user_id = u.id
        WHERE p.id = ?
    """,
        (post_id,),
        one=True,
    )

    if not post_raw:
        # The abort function will stop the request and show a 404 Not Found page.
        abort(404)

    #  Moderation for the Main Post
    # Convert the raw database row to a mutable dictionary
    post = dict(post_raw)
    # Unpack the tuple from moderate_content, we only need the moderated content string here
    moderated_post_content, _ = moderate_content(post["content"])
    post["content"] = moderated_post_content

    # if repost, return the original post, too
    original_post = None
    if post["is_repost"] == True:
        original_post_id = post["original_post"]
        original_post = query_db(
            """
            SELECT p.id as id, p.content, p.created_at, u.username, u.id as user_id
            FROM posts p
            JOIN users u ON p.user_id = u.id
            WHERE p.id = ?
            """,
            (original_post_id,),
            one=True,
        )

    #  Fetch Reactions (No moderation needed)
    reactions = query_db(
        """
        SELECT reaction_type, COUNT(*) as count
        FROM reactions
        WHERE post_id = ?
        GROUP BY reaction_type
    """,
        (post_id,),
    )

    #  Fetch and Moderate Comments
    comments_raw = query_db("SELECT c.id, c.content, c.created_at, u.username, u.id as user_id FROM comments c JOIN users u ON c.user_id = u.id WHERE c.post_id = ? ORDER BY c.created_at ASC", (post_id,))

    comments = []  # Create a new list for the moderated comments
    for comment_raw in comments_raw:
        comment = dict(comment_raw)  # Convert to a dictionary
        # Moderate the content of each comment
        print(comment["content"])
        moderated_comment_content, _ = moderate_content(comment["content"])
        comment["content"] = moderated_comment_content
        comments.append(comment)

    # Pass the moderated data to the template
    return render_template("post_detail.html.j2", post=post, reactions=reactions, comments=comments, reaction_emojis=REACTION_EMOJIS, reaction_types=REACTION_TYPES, original_post=original_post)


@app.route("/about")
def about():
    return render_template("about.html.j2")


@app.route("/privacy")
def privacy():
    return render_template("privacy.html.j2")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        location = request.form.get("location", "")
        birthdate = request.form.get("birthdate", "")
        profile = request.form.get("profile", "")

        hashed_password = generate_password_hash(password)

        db = get_db()
        cur = db.cursor()
        try:
            cur.execute("INSERT INTO users (username, password, location, birthdate, profile) VALUES (?, ?, ?, ?, ?)", (username, hashed_password, location, birthdate, profile))
            db.commit()

            # 1. Get the ID of the user we just created.
            new_user_id = cur.lastrowid

            # 2. Add user info to the session cookie.
            session.clear()  # Clear any old session data
            session["user_id"] = new_user_id
            session["username"] = username

            # 3. Flash a welcome message and redirect to the feed.
            flash(f"Welcome, {username}! Your account has been created.", "success")
            return redirect(url_for("feed"))  # Redirect to the main feed/dashboard

        except sqlite3.IntegrityError:
            flash("Username already taken. Please choose another one.", "danger")
        finally:
            cur.close()
            db.close()

    return render_template("signup.html.j2")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        db = get_db()
        user = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        db.close()

        # 1. Check if the user exists.
        # 2. If user exists, use check_password_hash to securely compare the password.
        #    This function handles the salt and prevents timing attacks.
        if user and check_password_hash(user["password"], password):
            # Password is correct!
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            flash("Logged in successfully.", "success")
            return redirect(url_for("feed"))
        else:
            # User does not exist or password was incorrect.
            flash("Invalid username or password.", "danger")

    return render_template("login.html.j2")


@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out.", "info")
    return redirect(url_for("login"))


@app.route("/posts/<int:post_id>/comment", methods=["POST"])
def add_comment(post_id):
    """Handles adding a new comment to a specific post."""
    user_id = session.get("user_id")

    # Block access if user is not logged in
    if not user_id:
        flash("You must be logged in to comment.", "danger")
        return redirect(url_for("login"))

    # Get content from the submitted form
    content = request.form.get("content")

    # Basic validation to ensure comment is not empty
    if content and content.strip():
        db = get_db()
        db.execute("INSERT INTO comments (post_id, user_id, content) VALUES (?, ?, ?)", (post_id, user_id, content))
        db.commit()
        flash("Your comment was added.", "success")
    else:
        flash("Comment cannot be empty.", "warning")

    # Redirect back to the page the user came from (likely the post detail page)
    return redirect(request.referrer or url_for("post_detail", post_id=post_id))


@app.route("/posts/<int:post_id>/repost", methods=["POST"])
def add_repost(post_id):
    """Handles adding a new repost to a specific post."""
    user_id = session.get("user_id")

    # Block access if user is not logged in
    if not user_id:
        flash("You must be logged in to repost.", "danger")
        return redirect(url_for("login"))

    # Get content from the submitted form
    content = request.form.get("content")

    # Basic validation to ensure repost is not empty
    db = get_db()
    cursor = db.execute("INSERT INTO posts (user_id, content, is_repost, original_post) VALUES (?, ?, ?, ?)", (user_id, content, 1, post_id))
    new_post_id = cursor.lastrowid
    db.commit()
    flash("Your repost was successfully created!", "success")

    # Redirect back to the page the user came from (likely the post detail page)
    return redirect(request.referrer or url_for("post_detail", post_id=new_post_id))


@app.route("/comments/<int:comment_id>/delete", methods=["POST"])
def delete_comment(comment_id):
    """Handles deleting a comment."""
    user_id = session.get("user_id")

    # Block access if user is not logged in
    if not user_id:
        flash("You must be logged in to delete a comment.", "danger")
        return redirect(url_for("login"))

    # Find the comment and the original post's author ID
    comment = query_db(
        """
        SELECT c.id, c.user_id, p.user_id as post_author_id
        FROM comments c
        JOIN posts p ON c.post_id = p.id
        WHERE c.id = ?
    """,
        (comment_id,),
        one=True,
    )

    # Check if the comment exists
    if not comment:
        flash("Comment not found.", "danger")
        return redirect(request.referrer or url_for("feed"))

    # Security Check: Allow deletion if the user is the comment's author OR the post's author
    if user_id != comment["user_id"] and user_id != comment["post_author_id"]:
        flash("You do not have permission to delete this comment.", "danger")
        return redirect(request.referrer or url_for("feed"))

    # If all checks pass, proceed with deletion
    db = get_db()
    db.execute("DELETE FROM comments WHERE id = ?", (comment_id,))
    db.commit()

    flash("Comment successfully deleted.", "success")
    # Redirect back to the page the user came from
    return redirect(request.referrer or url_for("feed"))


@app.route("/react", methods=["POST"])
def add_reaction():
    """Handles adding a new reaction or updating an existing one."""
    user_id = session.get("user_id")

    if not user_id:
        flash("You must be logged in to react.", "danger")
        return redirect(url_for("login"))

    post_id = request.form.get("post_id")
    new_reaction_type = request.form.get("reaction")

    if not post_id or not new_reaction_type:
        flash("Invalid reaction request.", "warning")
        return redirect(request.referrer or url_for("feed"))

    db = get_db()

    # Step 1: Check if a reaction from this user already exists on this post.
    existing_reaction = query_db("SELECT id FROM reactions WHERE post_id = ? AND user_id = ?", (post_id, user_id), one=True)

    if existing_reaction:
        # Step 2: If it exists, UPDATE the reaction_type.
        db.execute("UPDATE reactions SET reaction_type = ? WHERE id = ?", (new_reaction_type, existing_reaction["id"]))
    else:
        # Step 3: If it does not exist, INSERT a new reaction.
        db.execute("INSERT INTO reactions (post_id, user_id, reaction_type) VALUES (?, ?, ?)", (post_id, user_id, new_reaction_type))

    db.commit()

    return redirect(request.referrer or url_for("feed"))


@app.route("/unreact", methods=["POST"])
def unreact():
    """Handles removing a user's reaction from a post."""
    user_id = session.get("user_id")

    if not user_id:
        flash("You must be logged in to unreact.", "danger")
        return redirect(url_for("login"))

    post_id = request.form.get("post_id")

    if not post_id:
        flash("Invalid unreact request.", "warning")
        return redirect(request.referrer or url_for("feed"))

    db = get_db()

    # Remove the reaction if it exists
    existing_reaction = query_db("SELECT id FROM reactions WHERE post_id = ? AND user_id = ?", (post_id, user_id), one=True)

    if existing_reaction:
        db.execute("DELETE FROM reactions WHERE id = ?", (existing_reaction["id"],))
        db.commit()
        flash("Reaction removed.", "success")
    else:
        flash("No reaction to remove.", "info")

    return redirect(request.referrer or url_for("feed"))


@app.route("/u/<int:user_id>/follow", methods=["POST"])
def follow_user(user_id):
    """Handles the logic for the current user to follow another user."""
    follower_id = session.get("user_id")

    # Security: Ensure user is logged in
    if not follower_id:
        flash("You must be logged in to follow users.", "danger")
        return redirect(url_for("login"))

    # Security: Prevent users from following themselves
    if follower_id == user_id:
        flash("You cannot follow yourself.", "warning")
        return redirect(request.referrer or url_for("feed"))

    # Check if the user to be followed actually exists
    user_to_follow = query_db("SELECT id FROM users WHERE id = ?", (user_id,), one=True)
    if not user_to_follow:
        flash("The user you are trying to follow does not exist.", "danger")
        return redirect(request.referrer or url_for("feed"))

    db = get_db()
    try:
        # Insert the follow relationship. The PRIMARY KEY constraint will prevent duplicates if you've set one.
        db.execute("INSERT INTO follows (follower_id, followed_id) VALUES (?, ?)", (follower_id, user_id))
        db.commit()
        username_to_follow = query_db("SELECT username FROM users WHERE id = ?", (user_id,), one=True)["username"]
        flash(f"You are now following {username_to_follow}.", "success")
    except sqlite3.IntegrityError:
        flash("You are already following this user.", "info")

    return redirect(request.referrer or url_for("feed"))


@app.route("/u/<int:user_id>/unfollow", methods=["POST"])
def unfollow_user(user_id):
    """Handles the logic for the current user to unfollow another user."""
    follower_id = session.get("user_id")

    # Security: Ensure user is logged in
    if not follower_id:
        flash("You must be logged in to unfollow users.", "danger")
        return redirect(url_for("login"))

    db = get_db()
    cur = db.execute("DELETE FROM follows WHERE follower_id = ? AND followed_id = ?", (follower_id, user_id))
    db.commit()

    if cur.rowcount > 0:
        # cur.rowcount tells us if a row was actually deleted
        username_unfollowed = query_db("SELECT username FROM users WHERE id = ?", (user_id,), one=True)["username"]
        flash(f"You have unfollowed {username_unfollowed}.", "success")
    else:
        # This case handles if someone tries to unfollow a user they weren't following
        flash("You were not following this user.", "info")

    # Redirect back to the page the user came from
    return redirect(request.referrer or url_for("feed"))


@app.route("/admin")
def admin_dashboard():
    """Displays the admin dashboard with users, posts, and comments, sorted by risk."""

    if session.get("username") != "admin":
        flash("You do not have permission to access this page.", "danger")
        return redirect(url_for("feed"))

    RISK_LEVELS = {"HIGH": 5, "MEDIUM": 3, "LOW": 1}
    PAGE_SIZE = 50

    def get_risk_profile(score):
        if score >= RISK_LEVELS["HIGH"]:
            return "HIGH", 3
        elif score >= RISK_LEVELS["MEDIUM"]:
            return "MEDIUM", 2
        elif score >= RISK_LEVELS["LOW"]:
            return "LOW", 1
        return "NONE", 0

    # Get pagination and current tab parameters
    try:
        users_page = int(request.args.get("users_page", 1))
        posts_page = int(request.args.get("posts_page", 1))
        comments_page = int(request.args.get("comments_page", 1))
    except ValueError:
        users_page = 1
        posts_page = 1
        comments_page = 1

    current_tab = request.args.get("tab", "users")  # Default to 'users' tab

    users_offset = (users_page - 1) * PAGE_SIZE

    # First, get all users to calculate risk, then apply pagination in Python
    # It's more complex to do this efficiently in SQL if risk calc is Python-side
    all_users_raw = query_db("SELECT id, username, profile, created_at FROM users")
    all_users = []
    for user in all_users_raw:
        user_dict = dict(user)
        user_risk_score = user_risk_analysis(user_dict["id"])
        risk_label, risk_sort_key = get_risk_profile(user_risk_score)
        user_dict["risk_label"] = risk_label
        user_dict["risk_sort_key"] = risk_sort_key
        user_dict["risk_score"] = min(5.0, round(user_risk_score, 2))
        all_users.append(user_dict)

    all_users.sort(key=lambda x: x["risk_score"], reverse=True)
    total_users = len(all_users)
    users = all_users[users_offset : users_offset + PAGE_SIZE]
    total_users_pages = (total_users + PAGE_SIZE - 1) // PAGE_SIZE

    # --- Posts Tab Data ---
    posts_offset = (posts_page - 1) * PAGE_SIZE
    total_posts_count = query_db("SELECT COUNT(*) as count FROM posts", one=True)["count"]
    total_posts_pages = (total_posts_count + PAGE_SIZE - 1) // PAGE_SIZE

    posts_raw = query_db(
        f"""
        SELECT p.id, p.content, p.created_at, u.username, u.created_at as user_created_at
        FROM posts p JOIN users u ON p.user_id = u.id
        ORDER BY p.id DESC -- Order by ID for consistent pagination before risk sort
        LIMIT ? OFFSET ?
    """,
        (PAGE_SIZE, posts_offset),
    )
    posts = []
    for post in posts_raw:
        post_dict = dict(post)
        _, base_score = moderate_content(post_dict["content"])
        final_score = base_score
        author_created_dt = post_dict["user_created_at"]
        author_age_days = (datetime.utcnow() - author_created_dt).days
        if author_age_days < 7:
            final_score *= 1.5
        risk_label, risk_sort_key = get_risk_profile(final_score)
        post_dict["risk_label"] = risk_label
        post_dict["risk_sort_key"] = risk_sort_key
        post_dict["risk_score"] = round(final_score, 2)
        posts.append(post_dict)

    posts.sort(key=lambda x: x["risk_score"], reverse=True)  # Sort after fetching and scoring

    # --- Comments Tab Data ---
    comments_offset = (comments_page - 1) * PAGE_SIZE
    total_comments_count = query_db("SELECT COUNT(*) as count FROM comments", one=True)["count"]
    total_comments_pages = (total_comments_count + PAGE_SIZE - 1) // PAGE_SIZE

    comments_raw = query_db(
        f"""
        SELECT c.id, c.content, c.created_at, u.username, u.created_at as user_created_at
        FROM comments c JOIN users u ON c.user_id = u.id
        ORDER BY c.id DESC -- Order by ID for consistent pagination before risk sort
        LIMIT ? OFFSET ?
    """,
        (PAGE_SIZE, comments_offset),
    )
    comments = []
    for comment in comments_raw:
        comment_dict = dict(comment)
        _, score = moderate_content(comment_dict["content"])
        author_created_dt = comment_dict["user_created_at"]
        author_age_days = (datetime.utcnow() - author_created_dt).days
        if author_age_days < 7:
            score *= 1.5
        risk_label, risk_sort_key = get_risk_profile(score)
        comment_dict["risk_label"] = risk_label
        comment_dict["risk_sort_key"] = risk_sort_key
        comment_dict["risk_score"] = round(score, 2)
        comments.append(comment_dict)

    comments.sort(key=lambda x: x["risk_score"], reverse=True)  # Sort after fetching and scoring

    return render_template(
        "admin.html.j2",
        users=users,
        posts=posts,
        comments=comments,
        # Pagination for Users
        users_page=users_page,
        total_users_pages=total_users_pages,
        users_has_next=(users_page < total_users_pages),
        users_has_prev=(users_page > 1),
        # Pagination for Posts
        posts_page=posts_page,
        total_posts_pages=total_posts_pages,
        posts_has_next=(posts_page < total_posts_pages),
        posts_has_prev=(posts_page > 1),
        # Pagination for Comments
        comments_page=comments_page,
        total_comments_pages=total_comments_pages,
        comments_has_next=(comments_page < total_comments_pages),
        comments_has_prev=(comments_page > 1),
        current_tab=current_tab,
        PAGE_SIZE=PAGE_SIZE,
    )


@app.route("/admin/delete/user/<int:user_id>", methods=["POST"])
def admin_delete_user(user_id):
    if session.get("username") != "admin":
        flash("You do not have permission to perform this action.", "danger")
        return redirect(url_for("feed"))

    if user_id == session.get("user_id"):
        flash("You cannot delete your own account from the admin panel.", "danger")
        return redirect(url_for("admin_dashboard"))

    db = get_db()
    db.execute("DELETE FROM users WHERE id = ?", (user_id,))
    db.commit()
    flash(f"User {user_id} and all their content has been deleted.", "success")
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/delete/post/<int:post_id>", methods=["POST"])
def admin_delete_post(post_id):
    if session.get("username") != "admin":
        flash("You do not have permission to perform this action.", "danger")
        return redirect(url_for("feed"))

    db = get_db()
    db.execute("DELETE FROM comments WHERE post_id = ?", (post_id,))
    db.execute("DELETE FROM reactions WHERE post_id = ?", (post_id,))
    db.execute("DELETE FROM posts WHERE id = ?", (post_id,))
    db.commit()
    flash(f"Post {post_id} has been deleted.", "success")
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/delete/comment/<int:comment_id>", methods=["POST"])
def admin_delete_comment(comment_id):
    if session.get("username") != "admin":
        flash("You do not have permission to perform this action.", "danger")
        return redirect(url_for("feed"))

    db = get_db()
    db.execute("DELETE FROM comments WHERE id = ?", (comment_id,))
    db.commit()
    flash(f"Comment {comment_id} has been deleted.", "success")
    return redirect(url_for("admin_dashboard"))


@app.route("/rules")
def rules():
    return render_template("rules.html.j2")


@app.template_global()
def loop_color(user_id):
    # Generate a pastel color based on user_id hash
    h = hashlib.md5(str(user_id).encode()).hexdigest()
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return f"rgb({r % 128 + 80}, {g % 128 + 80}, {b % 128 + 80})"


# ----- Functions to be implemented are below


# Task 3.1
def recommend(user_id, filter_following):
    """
    Args:
        user_id: The ID of the current user.
        filter_following: Boolean, True if we only want to see recommendations from followed users.

    Returns:
        A list of 5 recommended posts, in reverse-chronological order.

    To test whether your recommendation algorithm works, let's pretend we like the DIY topic. Here are some users
    that often post DIY comment and a few example posts. Make sure your account did not engage with anything else.
    You should test your algorithm with these and see if your recommendation algorithm picks up on your interest in
    DIY and starts showing related content.

    Users: @starboy99, @DancingDolphin, @blogger_bob
    Posts: 1810, 1875, 1880, 2113

    Materials:
    - https://www.nvidia.com/en-us/glossary/recommendation-system/
    - http://www.configworks.com/mz/handout_recsys_sac2010.pdf
    - https://www.researchgate.net/publication/227268858_Recommender_Systems_Handbook
    """
    if filter_following:
        return {}

    # build a sparse matrix according to users' likes ################################
    query = """
    SELECT user_id, post_id, reaction_type
    FROM reactions
    WHERE reaction_type IN ('haha', 'like', 'wow', 'love');
    """  # (user, post), ifLiked
    users_2_liked_posts = query_db(query)

    # appeared users and posts
    users_set = {row[0] for row in users_2_liked_posts}
    posts_set = {row[1] for row in users_2_liked_posts}
    users_appeared = sorted(users_set)
    posts_appeared = sorted(posts_set)

    # for sparse matrix
    user_to_index = {}  # user_id: (123,45,67)->(0,1)
    post_to_index = {}
    for idx, user_id in enumerate(users_appeared):
        user_to_index[user_id] = idx
    for idx, post_id in enumerate(posts_appeared):
        post_to_index[post_id] = idx

    # for sparse matrix
    rows = []
    cols = []
    data = []

    for user_2_liked_post in users_2_liked_posts:
        user_index = user_to_index[user_2_liked_post[0]]
        post_index = post_to_index[user_2_liked_post[1]]
        rows.append(user_index)
        cols.append(post_index)
        data.append(1)  # value == 1 meaning 'liked'

    user_like_post_matrix = csr_matrix((data, (rows, cols)), shape=(len(users_appeared), len(posts_appeared)))

    # find people with similar interests OR followed by user ################################

    # if current user never liked any posts, refer to followers only
    top_similar_users_index = []
    current_user_never_liked = False
    if user_id not in user_to_index:
        current_user_never_liked = True
    else:
        user_index = user_to_index[user_id]

        # calculate similarity using cosine similarity, for all
        similarity_matrix = cosine_similarity(user_like_post_matrix)
        similarity_scores = similarity_matrix[user_index]

        # top 10 most similar users, not including this user self
        all_similar_index = similarity_scores.argsort()[-11:][::-1].tolist()  # desc, top11(including self)
        for index in all_similar_index:
            if index != user_index:
                top_similar_users_index.append(index)
            if len(top_similar_users_index) >= 10:
                break

    # followed users
    followed_users = query_db("SELECT followed_id FROM follows WHERE follower_id = ?", (user_id,))
    followed_user_id = [user["followed_id"] for user in followed_users]

    # followed users -> matrix index, if they've liked something
    followed_users_index = [user_to_index[i] for i in followed_user_id if i in user_to_index]
    reference_users = list(set(top_similar_users_index) | set(followed_users_index))  # similar users + followed users, each have 1 vote

    # find posts that were liked by those people while haven't been liked by this user ################################

    liked_by_given_user = set()  # posts already liked by the given user
    if not current_user_never_liked:
        liked_by_given_user = user_like_post_matrix[user_index].nonzero()[1]

    posts_2_likes = {post: 0 for post in posts_appeared}  # to find most liked posts by reference users

    for user_index in reference_users:
        # posts liked
        liked_by_user = user_like_post_matrix[user_index].nonzero()[1]

        for post_index in liked_by_user:
            if post_index not in liked_by_given_user:
                posts_2_likes[posts_appeared[post_index]] += 1

    sorted_posts = sorted(posts_2_likes.items(), key=lambda x: x[1], reverse=True)  # sort the posts
    recommended_post_id = [post for post, count in sorted_posts[:5]]  # top 5 posts

    # structured result for returning
    where_clause = "WHERE p.id IN ("
    for id in recommended_post_id:
        where_clause += str(id) + ","
    where_clause = where_clause[:-1] + ")"  # delete comma

    query = f"""
        SELECT p.id, p.content, p.created_at, u.username, u.id as user_id, is_repost, original_post
        FROM posts p
        JOIN users u ON p.user_id = u.id
        {where_clause}
        ORDER BY p.created_at DESC
    """
    print(query)
    recommended_posts = query_db(query)

    return recommended_posts


# Task 3.2
def user_risk_analysis(user_id):
    """
    Args:
        user_id: The ID of the user on which we perform risk analysis.

    Returns:
        A float number score showing the risk associated with this user. There are no strict rules or bounds to this score,
        other than that a score of less than 1.0 means no risk, 1.0 to 3.0 is low risk, 3.0 to 5.0 is medium risk and above
        5.0 is high risk. (An upper bound of 5.0 is applied to this score elsewhere in the codebase)

        You will be able to check the scores by logging in with the administrator account:
            username: admin
            password: admin
        Then, navigate to the /admin endpoint. (http://localhost:8080/admin)
    """

    score = 0.0

    # user profile
    user = query_db("SELECT profile, created_at FROM users WHERE id = ?", (user_id,))
    user_profile = user[0]["profile"]
    user_created_at = user[0]["created_at"]
    _, profile_score = moderate_content(user_profile)

    # posts
    posts = query_db("SELECT * FROM posts WHERE user_id = ?", (user_id,))

    average_post_score = 0
    if posts:  # if not, this value is 0
        for post in posts:
            _, content_score = moderate_content(post["content"])
            average_post_score += content_score
        average_post_score /= len(posts)

    # comments
    comments = query_db("SELECT * FROM comments WHERE user_id = ?", (user_id,))

    average_comment_score = 0
    if comments:  # if not, this value is 0
        for comment in comments:
            _, content_score = moderate_content(comment["content"])
            average_comment_score += content_score
        average_comment_score /= len(comments)

    # formula
    content_risk_score = (profile_score * 1) + (average_post_score * 3) + (average_comment_score * 1)

    # repeated content
    contents_2_counts = query_db(
        "SELECT content, COUNT(*) as content_count\
        FROM posts\
        WHERE user_id = ?\
        GROUP BY content\
        ORDER BY content_count DESC",
        (user_id,),
    )

    if contents_2_counts:
        if contents_2_counts[0]["content_count"] >= 3:
            content_risk_score += 2.0

    # account age
    user_risk_score = content_risk_score
    user_age_days = (datetime.utcnow() - user_created_at).days
    if user_age_days < 7:
        user_risk_score *= 1.5
    elif user_age_days < 30:
        user_risk_score *= 1.2

    # return user_risk_score
    return min(user_risk_score, 5.0)


# Task 3.3
def moderate_content(content):
    """
    Args
        content: the text content of a post or comment to be moderated.

    Returns:
        A tuple containing the moderated content (string) and a severity score (float). There are no strict rules or bounds to the severity score, other than that a score of less than 1.0 means no risk, 1.0 to 3.0 is low risk, 3.0 to 5.0 is medium risk and above 5.0 is high risk.

    This function moderates a string of content and calculates a severity score based on
    rules loaded from the 'censorship.dat' file. These are already loaded as TIER1_WORDS, TIER2_PHRASES and TIER3_WORDS. Tier 1 corresponds to strong profanity, Tier 2 to scam/spam phrases and Tier 3 to mild profanity.

    You will be able to check the scores by logging in with the administrator account:
            username: admin
            password: admin
    Then, navigate to the /admin endpoint. (http://localhost:8080/admin)
    """
    if content == None:
        return None, 0.0

    # severe ########################################################
    # tier1 word ---------------------------------------------------------
    TIER1_PATTERN = r"\b(" + "|".join(TIER1_WORDS) + r")\b"  # a regex pattern that matches whole word in tier 1 list
    matches = re.findall(TIER1_PATTERN, content, flags=re.IGNORECASE)  # find all the matching words
    if len(matches) > 0:
        return "[content removed due to severe violation]", 5.0

    # tier2 phrase ---------------------------------------------------------
    TIER2_PATTERN = r"(" + "|".join(TIER2_PHRASES) + r")"  # a regex pattern that matches phrases in tier 2 list
    matches = re.findall(TIER2_PATTERN, content, flags=re.IGNORECASE)  # find all the matching phrases
    if len(matches) > 0:
        return "[content removed due to spam/scam policy]", 5.0

    # scored violations & filtering ########################################################
    moderated_content = ""
    score = 0.0

    # tier 3 word ---------------------------------------------------------
    TIER3_PATTERN = r"\b(" + "|".join(TIER3_WORDS) + r")\b"  # a regex pattern that matches whole word in tier 3 list
    matches = re.findall(TIER3_PATTERN, content, flags=re.IGNORECASE)  # find all the matching words
    score += len(matches) * 2
    moderated_content = re.sub(TIER3_PATTERN, lambda m: "*" * len(m.group(0)), content, flags=re.IGNORECASE)  # replace all words with *

    # external links ---------------------------------------------------------
    # original regex string comes from https://stackoverflow.com/questions/3809401/what-is-a-good-regular-expression-to-match-a-url
    # https? or ftp urls can be detected.
    # not including internal link, have to use 'localhost' cause I don't actually have a domain name
    domain_name = "localhost"
    domain_escaped = re.escape(domain_name)  # domain name often look like 'example.com'

    # when using 'rf', every {} except those containing vars, need to be {{}}
    EX_LINK_PATTERN = rf"(?:https?|ftp):\/\/(?!(?:www\.)?{domain_escaped}\b)(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{{1,256}}(?:\.[a-zA-Z0-9()]{{1,6}}\b|\:\d{{1,5}})(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)"
    matches = re.findall(EX_LINK_PATTERN, moderated_content, flags=re.IGNORECASE)  # find all the matching urls
    score += len(matches) * 2
    moderated_content = re.sub(EX_LINK_PATTERN, "[link removed]", moderated_content, flags=re.IGNORECASE)  # replace all urls

    # excessive capitalization ---------------------------------------------------------
    total_letter_count = len(re.findall(r"[a-zA-Z]", content))  # total
    if total_letter_count > 15:
        uppercase_letter_count = len(re.findall(r"[A-Z]", content))  # upper
        if uppercase_letter_count / total_letter_count > 0.7:
            score += 0.5

    # one more ########################################################
    # detect special character ratio
    SPECIAL_CHARACTER_PATTERN = r'[^a-zA-Z0-9,.?!:;\'" -]'
    matches = re.findall(SPECIAL_CHARACTER_PATTERN, content)
    special_character_count = len(matches)

    NON_SPACE_PATTERN = r"\S"
    total_character_count = len(re.findall(NON_SPACE_PATTERN, moderated_content))  # length of characters, except " "

    if total_character_count > 15:
        special_character_ratio = special_character_count / total_character_count
        if special_character_ratio > 0.5:
            score += 2

    return moderated_content, score


if __name__ == "__main__":
    app.run(debug=True, port=8080)

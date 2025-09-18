# Mini Social

This is a mock social media platform created for the Social Computing course at the University of Oulu by Dr. Aku Visuri and Daniel Szabo.


The codebase contains two missing implementations: the moderation and the recommendation algorithm.

## Usage

1. Clone this repository onto your computer
2. Make sure you have Python 3.11 installed
3. Install dependencies by running `pip install -r requirements.txt`
4. Start the server by running `pyhon ./app.py`
5. The web application will be available at http://127.0.0.1:8080. 
6. The admin interface is at http://127.0.0.1:8080/admin, available only to the Admin user (username: admin, pw: admin)

### Docker

If you prefer to run Mini-Social as a docker container, you can build the image and run it as follows:

1. Build image with `docker build --tag mini-social .`
2. Run image with `docker run -d -p 8000:5000 mini-social` where 8000 is the port you will access the application at `http://127.0.0.1:8000`
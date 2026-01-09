python -m venv .venv
. .venv/bin/activate
sudo apt install -y python3-picamera2 python3-libcamera libcamera-apps
pip install -e .

mkdir weights
touch .env
echo "WEIGHTS=\"$(pwd)/weights\"" > .env
pip install gdown
gdown --folder https://drive.google.com/drive/folders/111NYTqKwKNPBgeXxntdD4WQ5FDEb6IkG -O weights
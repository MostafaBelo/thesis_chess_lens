python -m venv .venv
. .venv/bin/activate
pip install -e .

pip install gdown
mkdir weights
gdown --folder https://drive.google.com/drive/folders/111NYTqKwKNPBgeXxntdD4WQ5FDEb6IkG -O weights
touch .env
echo "WEIGHTS=\"$(pwd)/weights\"" > .env
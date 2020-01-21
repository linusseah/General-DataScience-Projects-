sudo yum install python3 python3-wheel -y
sudo pip3 install virtualenv
virtualenv flask_env3
source flask_env3/bin/activate
pip3 install -r ec2_requirements.txt
python -m nltk.downloader all
sudo yum install python3-tkinter -y
mkdir flask_env3/flask_app
scp Archive.zip flask_env3/flask_app/
cd flask_env3/flask_app/
unzip Archive.zip
python inside-out.py

#FROM ubuntu:23.10
FROM python:3.9.17
COPY . /digit/
#RUN apt-get update
#RUN apt-get install python3 python3-pip
WORKDIR /digit
CMD ["python3","plot_digits_classification.py"]
CMD ["pip3", "install" ,"-r", "requirement.txt"]
#RUN pip3 install -r ./digit/requirement.txt
#WORKDIR /digit
ENV FLASK_APP=api/digit_pred
CMD ["flask","run","--host=0.0.0.0"]
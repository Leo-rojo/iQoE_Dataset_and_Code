## iQoE web application
* Structure:
  * cssandjsandvideos should contain all the css, js, and videos files produced in the previous steps. A copy of the videofiles used can be downloaded from: https://drive.google.com/file/d/10PY3e37GCDD5DIgyq6Xmafr921MQkKqS/view?usp=sharing
  * templates should contain all the html files
  * original_database contains useful files for dividing the video pool of 1000 videos into train and test set and from train select randomly videos for baselines models

* How to run the web application:
  * Run 'Generate_initial_experiences.py' in order to fill original_database folder (used to extract and track train and test videos for each user)
  * Run 'app_v3.py' to run the web application. It contains all the backend logic, deployed with flask library.
  * It will run on localhost:7000 and for each user that starts the assessment a new folder called "user_unique_id" will be created

* How to run the web application in the apache server:
  * Follow the guide in https://flask.palletsprojects.com/en/2.0.x/deploying/mod_wsgi/

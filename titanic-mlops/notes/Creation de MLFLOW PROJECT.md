Creation de MLFLOW PROJECT : 





comment organiser un projet de machine learning avec MLflow ? 

* Le nom du projet (exemple : “validation\_dataset”)
* Le fichier qui décrit l’environnement (les dépendances Python à installer) c un fichier .yml
* Les points d’entrée : les scripts (Python ou Shell): C’est là où tu écris ce que doit faire ton projet (par exemple lire un fichier CSV, entraîner un modèle…).



exp de Fichier MLproject :

|<br />YAML|
|-|
|name: test\_mlflow<br />conda\_env: environment.yml<br />entry\_points:<br />  main:<br />    parameters:<br />      input\_csv: {type: str, default: "data.csv"}<br />    command: "python script.py --input\_csv {input\_csv}"|



Fichier environment.yml :

|<br />YAML|
|-|
|name: test\_mlflow\_env<br />channels:<br />&nbsp; - conda-forge<br />dependencies:<br />&nbsp; - python=3.8<br />&nbsp; - pandas<br />&nbsp; - mlflow|

&nbsp;en exécutant ce ce fichier avec la ligne de code : conda env create -f environment.yml , la mm environnement avec les mm libs et dependences seront installées partout ce qui assure que  le projet soit uctible 




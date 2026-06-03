pendant l'evaluation 
creer `data/raw` dans lequelle on unzip le fichier `vllm-0.10.1.zip`
```
bash
mkdir data
mkdir data/raw
unzip vllm-0.10.1.zip -d data/raw
```

unzipper le fichier `exams.zip` puis creer un DEUXIEME `data` dedans
```
bash
unzip exams.zip
mkdir exams_pkg/data
```

creer un dossier `datasets` dans `exams_pkg` dans lequel on unzip le `datasets_private.zip`
```
bash
mkdir exams_pkg/data/datasets
unzip datasets_private.zip -d exams_pkg/data/datasets
```

unzipper la moulinette a la racine (osef de celle la)
```
bash
unzip moulinette.zip
```

il ne reste plus qu'a lancer les commandes d'exams avec comme `--student-path` la racine du projet (`.`)
```
bash
RETRIEVAL:
./exams_pkg/exams/scripts/exam_retrieval.sh --student-path . --moulinette-path ./moulinette_pkg/moulinette-ubuntu
ou
ANSWER:
./exams_pkg/exams/scripts/exam_answer.sh --student-path . --moulinette-path ./moulinette_pkg/moulinette-ubuntu
ou
EDGE_CASES:
./exams_pkg/exams/scripts/exam_edge_cases.sh --student-path .
```

le tour est joue bon chance pour le reste
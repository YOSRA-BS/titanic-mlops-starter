&nbsp;TITANIC \_MLOPS\_STARTER 



1. Flake8  C un outil incontournable dans l'ecosysteme python  utilisé pour vérifier la qualité et le style de votre code. Il appartient à la catégorie des linters, des programmes qui analysent le code source pour détecter des erreurs potentielles, des problèmes de style ou des constructions suspectes sans exécuter le programme.



il combine 3 bibliothèques principales: 



| Composant       | Rôle principal                                                                                                    |

| :-------------- | :---------------------------------------------------------------------------------------------------------------- |

| \*\*PyFlakes\*\*    | Analyse les erreurs logiques (variables non définies, imports inutilisés, etc.) sans vérifier le style.           |

| \*\*pycodestyle\*\* | Vérifie que le code respecte la \[PEP 8](https://peps.python.org/pep-0008/), le guide de style officiel de Python. |

| \*\*McCabe\*\*      | Mesure la \*\*complexité cyclomatique\*\* pour identifier les fonctions trop complexes et difficiles à maintenir.     |





Pourquoi utiliser Flake8 ?

L'utilisation de Flake8 dans un projet présente plusieurs avantages majeurs :

Uniformité : Il garantit que tous les développeurs d'une équipe suivent les mêmes conventions de nommage et de mise en forme.

Prévention d'erreurs : Il détecte des fautes de frappe ou des erreurs de logique simples avant même que le code ne soit testé.

Lisibilité : En imposant des règles comme la longueur maximale des lignes ou l'espacement, il rend le code plus facile à lire pour les autres.





Installation et Usage

Il s'installe très simplement via pip :

Bash

pip install flake8

Pour analyser un fichier ou un dossier, il suffit de lancer la commande suivante dans votre terminal :

Bash

flake8 mon\_projet/



&nbsp;.flake8

Utilité globale

Rôle: Configurer l’outil flake8 qui vérifie la qualité/forme du code Python (style PEP8, erreurs courantes).

Pourquoi en MLOps:

Évite que du “mauvais” code parte en prod (mauvaise indentation, imports inutiles, etc.).

La CI (workflow ci.yml) appelle flake8 pour bloquer un commit si la qualité n’est pas OK.



1. requirements.txt

   Rôle: Liste des dépendances pour pip install -r requirements.txt.
   Usage: Utilisé dans les workflows GitHub (ci.yml, cd.yml, ct.yml) pour installer les libs avant d’exécuter tests/entraînement.







1. pytest.ini Configuration globale pour pytest
   setup.py : Utilité globale
   Rôle: Transformer ton projet en package installable (pip install .).
   Pourquoi en MLOps:

En CI/CD, tu peux installer ton projet comme une librairie dans un autre environnement (API, batch jobs…).










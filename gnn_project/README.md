# Deep Learning Project - Graph Neural Networks (GNN)

Ce repository est un projet de deep learning. Le but est d’introduire le concept de GNN (Graph Neural Networks). 

Il y a à disposition : 
- une vidéo d’introduction aux GNN : https://youtu.be/ks3VZtoXITw
- la présentation .pdf
- un code python correspondant à l'application d'un GCN à un ensemble de graphes moléculaires ZINC-250K.

Inspiré de l'article suivant : https://distill.pub/2021/gnn-intro/

## Implémentation python

Pour l'implémentation, différents éléments sont disponibles.

Les codes à exécuter :
- main_molecules_graph_regression.py est le code main : son exécution entraîne le modèle GCN
- plot_training_curves.py permet de plot les courbes d'entraînement (avec loss)
- visualize_molecules.py permet de visualiser un échantillon de molécules de train et test 

Les outils / résultats : 
- data contient les données. Normalement, en exécutant le main, on exécute un fichier qui télécharge la base de donnée source (1.3Go).
- les autres dossiers contiennents des outils ou des résultats (d'entraînement, de test)
- 4 plots en .png sont disponibles directement pour observer les résultats de l'entraînement du modèle
- un fichier result.txt contient les informations de fin d'epochage.

Je ne vous conseille pas de lancer le code. Le code à lancer est assez long (30min sur mon pc). 
Sinon, il faut exécuter le fichier python main_molecules_graph_regression.py, puis les autres codes (plot_training_curves.py, visualize_molecules.py) pour avoir la data visualisation.

Les packages suivants sont à installer en amont :

` pip install torch dgl numpy matplotlib tensorboardX tensorboard tqdm networkx `



Dans ce documents je vais tenter d'expliquer au mieux ma démarche.

Le code est accolé dans la pull request,
Mon pseudo est Neeko,
mon meilleur score est de 62.7%,
mon token de login 1ae455f73cbd42ea4d8cb1d98c1755ec38749392.

En ce qui concerne mon code j'ai tenté 2 grosses approches : 
- le RL comme on l'a vu en cours
- une tentative de reverse ingeneer le site


Mes premières observations : 
-Le problème est fortement cyclique et bruité, les taux de survie suivent des sinus avec des phases différentes pour chaque planète.
-Les méthodes basées sur l’observation locale seule (moving average) sont trop lentes pour réagir lorsque the Purge n'est plus exploitable avant 200 nouveau morty.
-La fréquence des sinus ne change pas, ce qui rend possible une approche prédictive si on peut estimer sa phase.


Pour la partie reverse ingeneer, j'ai tenté d'observer la distribution des taux de survie pour pouvoir les prédire et de décider plus intelligemment.

Pour ça j'ai d'abord réaliser un code pour dérouler 15 episode qui envoyer 1000 mortys sur chaques planètes (5 fois pour chaue planète), ça m'a permit de trouver les féquences des cosinus qui définissait les taux de survie. De manière moins absolue, ça m'a aussi permis d'obtenir l'amplitude moyenne.

A partir de ces informations, j'ai recréer en local un environnemen virtuel (que j'ai perdu sans faire exprès) où la phase était choisi aléatoirement pour pouvoir tester à large échelle mes algo et moyenner leur performances. 

J'ai pu tenter diverses approches et peut importe le nombre de trip réalisé, l'erreur de prédiction de la phase restait beaucoup trop importante. Après pas mal de recherche je n'ai pas vraiment réussi à créer d'algorithme profitant pleienement de la connaissance des courbes.

Une méthode intéressante que j'aurais voulu tester avec plus de temps aurait été de profiter du fait que cos(wt+O) = a x cos(wt) + B x cos(wt).
Cette idée de Victor aurait permis d'utliser des algo de RL pour trouver les A et B de chaques planètes.

N’ayant pas réussi à produire un algorithme parfait pour cette régression de phase, j’ai voulu tenter des approches plus “RL” :
Sachant que les données fluctuent beaucoup mais de manière cyclique et surtout bruité, j’ai testé des approches Thompson Sampling avec fenêtre glissante et filtre de Kalman pour suivre les phases.

Pour l’approche Thompson, je suis tombée dessus à plusieurs reprises durant mes recherches. Elle m’a semblé pertinente pour ses capacités à gérer l’exploration/exploitation de manière adaptative et probabiliste, en particulier dans un contexte où les performances des actions (planètes) fluctuent dans le temps.

Et pour l'utilisation des filtre de Kalman, c'est la capacité de prédiction de la phase du sinus dans un environement bruité qui m'intéressait.
Cela permet de prédire le taux de survie futur pour chaque planète,
D’adapter le nombre de Mortys envoyés (batch size) en fonction de la confiance sur la prédiction.
Et de combiner cette prédiction avec la stratégie de stickiness pour éviter des changements de planète trop fréquents.

Sur mon environnement virtuel, j'ai déjà atteint des taux de réussite autour des 73-74% mais c'est parce que je pouvais run 5000 essaies en moins de 20 secondes.
En pratique je n'ai réussi à avoir que 62.7% de réussite via sphinx, et de manière transparente avec une moyenne de résultat autour des 52-53%.
Not using noPV at all (except when it's resized into it)
-> Do the experiment (use 50% noPV, 100%...)

Used weighted loss because there's a lot of negative.

Weight formula #negative/#positive
0%noPV : 5.25
20%noPV : 5.75
100%noPV : 8.0
Any data to justify?

Adapt learning rate

Post-processing: 
- Need to find best threshold
  -> Need to have a validation set where you find the best threshold
- Would like to make borders straight (L1 loss)
- BCE (sans Loss)





Trouver des meilleurs noms pour les modeles (ou sinon faire un classement)
pour qu'on puisse les identifier facilement
J'ai besoin de ca pour faire les commentaires sur les resultats

Faire accuracy aussi?

Graphes precision-recall? F1?
Si jamais on n'a plus de place

Images + Predictions
Ce serait cool pour comparer sur la meme image, mais c'est pas tres scientifique
de faire des conclusions sur juste ca

Et puis ca prend beaucoup de place



Run chaque config plusieurs fois pour les stats
-> c'etait long et on nous avait pas reserve de la puissance

Full Adam ou un peu SGD?
Ca c'est important

Oversampling (utiliser plusieurs fois la meme image)
Avec 1000 ou 100 modeles ?

Utiliser IoU / Jaccard

On fait que des augmentations sur les PV (pas sur les noPV)

Clean code (modulariser)
Dire comment run
Jupyter notebooks pour tester

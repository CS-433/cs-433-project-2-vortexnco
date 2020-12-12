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

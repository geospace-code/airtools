python-AIRtools
===============

Port of P.C. Hansen's notable AIRtools Matlab suite of inversion / regularization tools

PC Hansen's ReguTools ports are also here.

Notes:

picard.py: The resulting graph does not exactly match Matlab results. This may originate from that 
with Numpy 1.8.1 the output of numpy.linalg.svd and scipy.linalg.svd do NOT exactly match Matlab.

maxent.py: Maximum Entropy Regularization

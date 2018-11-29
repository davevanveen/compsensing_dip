************************************************************************
            TVAL3 -- A New Efficient TV Solver for Compressive Sensing
************************************************************************

Copyright (C) 2009 Chengbo Li and Yin Zhang.




Introduction
====================
   
   TVAL3 refers to "TV optimization -- an ALternating minimization ALgorithm 
for Augmented Lagrangian functions", which is a TV based image reconstruction
and denosing solver.

TVAL3 aims at solving the ill-possed inverse problem: approximately 
recover image ubar from

                   f = A*ubar + omega,                              (1)

where ubar is the original signal/image, A is a measurement matrix, omega 
is addtive noise and f is the noisy observation of ubar. 

Given A and f, TVAL3 tries to recover ubar by solving TV regularization 
problems:

                     min_u 	TV(u).               		    (2)
		      s.t.  A*u = f-omega

          or         min_u      TV(u)+||Au-f||_2^2



Advantages
====================

1) TVAL3 accepts all kinds of matrices as the measurement matrix A as long as
   randomness is guaranteed. Normality or orthogonality is not required;

2) TVAL3 is extremely fast when A*x and A'*x can be defined in a fast way, 
   such as using WHT, FFT, or DCT with some randomness;
 
3) TVAL3 can handle complex signals and even complex measurement matrices.



How To Use  
====================

The solver is called in following ways:

               [U, out] = TVAL3(A,b,p,q,opts)
 	   or         U = TVAL3(A,b,p,q,opts).



Notice*:   Users should be aware that all fields of opts are assigned by default 
	   values, which are chosen based on authors' research or experience. 
	   However, at least one field of opts (any one) must be assigned by users.
	   

Please read more details in User's_Guild_for_TVAL3.pdf.      
   




Contact Information
=======================


This is only the test version of TVAL3. Please feel free to e-mail us with any
comments, bug reports, or suggestions. We are more than happy to hear that!

Chengbo Li	        cl9@rice.edu	        CAAM, Rice University	
Yin Zhang		yzhang@rice.edu		CAAM, Rice University
Wotao Yin               wotao.yin@rice.edu      CAAM, Rice University


Copyright Notice
====================

TVAL3 is free software, and you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software Foundation.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
PARTICULAR PURPOSE.  See the GNU General Public License for more details at
<http://www.gnu.org/licenses/>. 


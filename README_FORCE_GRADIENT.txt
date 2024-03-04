********************************************************************************

                        openQCD Simulation Programs

      extended version including Hessian-free force-gradient integrators

********************************************************************************

For general information, see the general README.


INTEGRATORS FOR MOLECULAR-DYNAMICS EQUATIIONS

This adapted version of the openQCD 2.4 code allows for nested hierarchical integrators for the molecular-dynamics equations, based on any combination of Hessian-free force-gradient integrators introduced in [Schaefers et al., 2024]. This also includes the non-gradient algorithms of Omelyan-Mryglod-Folk (OMF) up to 6th order. 

The current coefficient sets are minimizing the (weighted) norm of the leading error coefficients. The investigations have been made for Hessian-free force-gradient integrators on a single level, similar to the investigations made by Omelyan, Mryglod and Folk in 2003. For nested integrators, the optimization of the degrees of freedoms will most likely result in different coefficients and is part of future research.


AUTHORS

The initial release of the openQCD package was written by Martin Lüscher and
Stefan Schaefer. Support for Schrödinger functional boundary conditions was
added by John Bulava. Phase-periodic boundary conditions for the quark fields
were introduced by Isabel Campos and the implementation of the "exponential"
variant of the O(a) Pauli term was developed by Antonio Rago. Several modules
were taken over from the DD-HMC program tree, which includes contributions
from Luigi Del Debbio, Leonardo Giusti, Björn Leder and Filippo Palombi. 
This adapted version enabling the use of Hessian-free force-gradient integrators 
is based on version 2.4 of the openQCD package and was written by Jacob Finkenrath
and Kevin Schäfers. 


LICENSE

The software may be used under the terms of the GNU General Public Licence
(GPL).

BUG REPORTS

If a bug is discovered, please send a report to <schaefers@math.uni-wuppertal.de> or <finkenrath@Uni-wuppertal.de>.


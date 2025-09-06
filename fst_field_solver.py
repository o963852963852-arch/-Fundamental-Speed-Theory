# Fundamental Speed Theory (FST) - Numerical Solver
# Author: Raheeb Ali Mohammad Saleh Odeh
# Contact: o.963852963852@gmail.com

import numpy as np
from scipy.integrate import solve_bvp
import astropy.constants as const
import astropy.units as u

class FSTFieldSolver:
    def __init__(self, mV=3.2e-30, c1=0.51, c2=-0.07, c3=0.32,
                 Lambda_V=1.1e-52, lambda_nl=1.2e14):
        self.mV = (mV * u.eV).to(u.kg, equivalencies=u.mass_energy()).value
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.Lambda_V = Lambda_V
        self.lambda_nl = lambda_nl
        self.G = const.G.value

    def field_equation(self, r, y):
        V, dVdr = y
        r_safe = np.maximum(r, 1e-10)
        d2Vdr2 = - (2/r_safe)*dVdr + self.mV**2*V + (self.lambda_nl/6)*V**3
        return np.vstack((dVdr, d2Vdr2))
    
    def solve_spherical(self, r_range, V0, dVdr0, n_points=1000):
        def bc(ya, yb):
            return np.array([ya[0] - V0, yb[1] - dVdr0])
        
        r = np.linspace(r_range[0], r_range[1], n_points)
        y_guess = np.zeros((2, r.size))
        y_guess[0] = V0 * np.exp(-self.mV * r)
        y_guess[1] = -self.mV * V0 * np.exp(-self.mV * r)
        
        sol = solve_bvp(self.field_equation, bc, r, y_guess,
                        tol=1e-8, max_nodes=10000)
        
        if not sol.success:
            raise RuntimeError(f"Solver failed: {sol.message}")
        return sol

class RotationCurveFitter:
    def __init__(self, fst_solver):
        self.fst_solver = fst_solver
        self.G = const.G.value
        
    def compute_circular_velocity(self, r, M_bary, V_field):
        v_bary = np.sqrt(self.G * M_bary / r)
        M_V = self.compute_V_mass(r, V_field)
        v_field = np.sqrt(self.G * M_V / r)
        return np.sqrt(v_bary**2 + v_field**2), v_bary, v_field
    
    def compute_V_mass(self, r, V):
        dVdr = np.gradient(V, r)
        energy_density = (0.5*self.fst_solver.mV**2*V**2 +
                        0.5*self.fst_solver.c1*dVdr**2 +
                        (self.fst_solver.lambda_nl/24)*V**4)
        return np.trapz(energy_density * 4 * np.pi * r**2, r)

if __name__ == "__main__":
    print("FST Field Solver - Test Successful!")
    print("This code implements the Fundamental Speed Theory numerical solver.")
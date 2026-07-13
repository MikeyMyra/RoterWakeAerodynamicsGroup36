from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np

from plotter import plot
#use latex:
#plt.rcParams['text.usetex'] = True


class BEM:
    
    def __init__(self, J, radius, n_blades, U_inf):
        
        # Rotor specs (absolute values)
        self.radius = radius  # meters
        self.n_blades = n_blades
        self.blade_start_fraction = 0.25  # Fraction of radius (0-1)
        self.collective_blade_pitch = 46  # degrees
        self.collective_blade_pitch_location = 0.7  # Fraction of radius (0-1)
        
        # Operational specs
        self.U_inf = U_inf  # m/s
        self.J = J
        self.rpm = 60 * (self.U_inf / (J * 2 * self.radius))
        self.n_rps = self.rpm / 60      # rotational speed [rev/s] for propeller convention
        self.D = 2 * self.radius        # rotor diameter [m] for propeller convention
        self.altitude = 2000  # meters
        self.incidence = 0
        self.rotor_yaw = 0

        # Lifting-line vortex geometry / regularisation
        self.cp_chord_frac = 0.25      # control point at 1/4 chord, co-located with the bound vortex
        self.trail_chord_frac = 1.25   # on-blade trailing legs end 1/4 chord behind the TE

        # Rankine vortex-core radius [m], graded radially. The grid-scale ("sawtooth")
        # circulation mode is over-gained/ill-conditioned in the discrete solve, most
        # severely inboard where the chord is largest and the twist steepest (long,
        # inclined trailing legs coupling neighbouring panels). So we damp heavily near
        # the root and lightly near the tip, where a small core is needed to resolve the
        # circulation roll-off. Set self.vortex_core to a float to force a uniform core.
        self.vortex_core = None        # None -> use the graded root/tip values below
        self.vortex_core_root = 0.03   # core at the blade root [m] (heavy damping)
        self.vortex_core_tip = 0.006   # core at the blade tip  [m] (resolves the tip drop)

        # Airfoil data
        self.AoA, self.cl, self.cd, self.cm = self._get_airfoil()
        self.rho = self._get_isa_density(self.altitude)
    
    @staticmethod
    def _get_airfoil():
        
        data = []
        with open("assignment_2\\ARAD8pct_polar.txt", "r") as file:
            for line in file:
                row = line.strip().split()
                data.append(row)
        data = data[2:]
        
        AoA = [float(row[0]) for row in data]
        cl  = [float(row[1]) for row in data]
        cd  = [float(row[2]) for row in data]
        cm  = [float(row[3]) for row in data]
        
        return AoA, cl, cd, cm
    
    @staticmethod
    def _get_isa_density(h): 
        
        T0, p0, L, R, g = 288.15, 101325, 0.0065, 287.05, 9.80665
        
        T = T0 - L*h
        p = p0 * (T/T0)**(g/(R*L))
        rho = p/(R*T)
        
        return rho
    
    @staticmethod
    def _apply_glauert_correction(CT):
        
        CT1 = 1.816
        CT2 = (2 * np.sqrt(CT1)) - CT1
        
        if CT < 0:
            CT = 0
        
        if CT < CT2:
            a = (1/2) * (1 - np.sqrt(max(0, 1 - CT)))
        else:
            a = 1 + ((CT - CT1) / (4 * (np.sqrt(CT1) - 1)))
        
        return a

    @staticmethod
    def circulation_from_momentum(U_inf, omega, a, a_prime, n_blades):
        a = np.asarray(a, dtype=float)
        a_prime = np.asarray(a_prime, dtype=float)
        return (4 * np.pi * U_inf**2 * a * (1 + a)) / (n_blades * omega * (1 - a_prime))

    def _calculate_prandtl_factor(self, r_norm, a):
        
        mu = r_norm
        mu_root = self.blade_start_fraction
        
        if mu < mu_root:
            return 1.0
        
        if mu >= 1.00:
            return 0.0
        
        if a >= 1.00 or a <= 0.0:
            a = 0.5
        
        omega = (2 * np.pi * self.rpm) / 60
        lambda_local = (omega * r_norm * self.radius) / self.U_inf
        
        try:
            exponent_tip = -(self.n_blades / 2) * ((1 - mu) / mu) * np.sqrt(1 + (mu**2 * lambda_local**2) / ((1 + a)**2))
            #exponent_tip = np.clip(exponent_tip, -10, 0)
            f_tip = (2 / np.pi) * np.arccos(np.exp(exponent_tip))
        except:
            f_tip = 1.
        
        try:
            exponent_root = -(self.n_blades / 2) * ((mu - mu_root) / mu) * np.sqrt(1 + (mu**2 * lambda_local**2) / ((1 + a)**2))
            #exponent_root = np.clip(exponent_root, -10, 0)
            f_root = (2 / np.pi) * np.arccos(np.exp(exponent_root))
        except:
            f_root = 1.0
        
        F_total = f_tip * f_root
        #F_total = max(F_total, 0.0001)
        
        return F_total
    
    def _core_at_radius(self, r_abs):
        # Rankine core radius at a given radial station. Uniform if self.vortex_core is
        # set to a float; otherwise a linear ramp from vortex_core_root (blade start) to
        # vortex_core_tip (r = R). See __init__ for why the grading is needed.
        if self.vortex_core is not None:
            return self.vortex_core
        mu = r_abs / self.radius
        frac = np.clip((mu - self.blade_start_fraction) / (1.0 - self.blade_start_fraction), 0.0, 1.0)
        return self.vortex_core_root + (self.vortex_core_tip - self.vortex_core_root) * frac

    def biot_savart(self,X1,X2,Xp,gamma,rc=None):
        eps=1e-6
        R1=np.sqrt((Xp[0]-X1[0])**2+(Xp[1]-X1[1])**2+(Xp[2]-X1[2])**2)
        R1=max(R1,eps)
        R2=np.sqrt((Xp[0]-X2[0])**2+(Xp[1]-X2[1])**2+(Xp[2]-X2[2])**2)
        R2=max(R2,eps)
        R12x=(Xp[1]-X1[1])*(Xp[2]-X2[2])-(Xp[2]-X1[2])*(Xp[1]-X2[1])
        R12y=-(Xp[0]-X1[0])*(Xp[2]-X2[2])+(Xp[2]-X1[2])*(Xp[0]-X2[0])
        R12z=(Xp[0]-X1[0])*(Xp[1]-X2[1])-(Xp[1]-X1[1])*(Xp[0]-X2[0])
        R12sqrt=R12x**2+R12y**2+R12z**2

        # Graded core: key it to the filament's own radial station (distance from the
        # rotor axis = x). Applies to bound and wake filaments alike since a wake helix
        # stays at ~its shedding radius.
        if rc is None:
            rc=self._core_at_radius(0.5*(np.hypot(X1[1],X1[2])+np.hypot(X2[1],X2[2])))
        if rc>0.0:
            r0sq=(X2[0]-X1[0])**2+(X2[1]-X1[1])**2+(X2[2]-X1[2])**2
            R12sqrt=R12sqrt+(rc*rc)*r0sq
        R12sqrt=max(R12sqrt,eps)
        R01=(X2[0]-X1[0])*(Xp[0]-X1[0])+(X2[1]-X1[1])*(Xp[1]-X1[1])+(X2[2]-X1[2])*(Xp[2]-X1[2])
        R02=(X2[0]-X1[0])*(Xp[0]-X2[0])+(X2[1]-X1[1])*(Xp[1]-X2[1])+(X2[2]-X1[2])*(Xp[2]-X2[2])
        K=1/(4*np.pi*R12sqrt)*(R01/R1-R02/R2)
        C_ind=[float(K*(R12x)),float(K*(R12y)),float(K*(R12z))]
        U_ind=[float(K*(R12x))*gamma,float(K*(R12y))*gamma,float(K*(R12z))*gamma]
        # if U_ind[0]==np.nan:
        #     U_ind=[0,0,0]
        # np.where(U_ind==np.nan,0)
        
        return C_ind,U_ind
    
    
    def calc_ind_filiment(self,Xp,r,dt=0.1,tend=5,plot=False):

        # self.yarr=r*np.sin(omega*self.tlst)
        # self.zarr=r*np.cos(omega*self.tlst)
        
        
        C_ind,uvwind=self.biot_savart([0,0,r-self.dr_used/2],[0,0,r+self.dr_used/2],Xp,1)

        uind=[uvwind[0]]
        vind=[uvwind[1]]
        wind=[uvwind[2]]
        cuind=[C_ind[0]]
        cvind=[C_ind[1]]
        cwind=[C_ind[2]]
        # C_ind,uvwind=self.biot_savart([0,0,r-self.dr_used],[0,0,r+self.dr_used],Xp,1)

        for ij in range(len(self.tlst)-1):
            C_ind,uvwind=self.biot_savart([self.xarr[ij],self.yarr[ij],self.zarr[ij]],[self.xarr[ij+1],self.yarr[ij+1],self.zarr[ij+1]],Xp,1)
            
            uind.append(uvwind[0])
            vind.append(uvwind[1])
            wind.append(uvwind[2])
            cuind.append(C_ind[0])
            cvind.append(C_ind[1])
            cwind.append(C_ind[2])
            C_ind,uvwind=self.biot_savart([self.xarr[ij+1],self.yarr2[ij+1],self.zarr2[ij+1]],[self.xarr[ij],self.yarr2[ij],self.zarr2[ij]],Xp,1)

            uind.append(uvwind[0])
            vind.append(uvwind[1])
            wind.append(uvwind[2])
            cuind.append(C_ind[0])
            cvind.append(C_ind[1])
            cwind.append(C_ind[2])

        
        # uind=[]

        if plot==True:
            fig=plt.figure()
            fig2=plt.figure()
            ax=fig.subplots(2,2)
            self.ax2=fig2.add_subplot(projection='3d')
            # self.ax2.plot([0,0],[0,0],[r-self.dr_used/2,r+self.dr_used/2],color='tab:blue')

            for ij in range(len(self.tlst)-1):
                
                # uind.append(self.biot_savart([self.xarr[ij],self.yarr[ij],self.zarr[ij]],[self.xarr[ij+1],self.yarr[ij+1],self.zarr[ij+1]],Xp,1)[0])
                ax[0,0].plot([self.xarr[ij],self.xarr[ij+1]],[self.yarr[ij],self.yarr[ij+1]])
                ax[0,1].plot([self.xarr[ij],self.xarr[ij+1]],[self.zarr[ij],self.zarr[ij+1]])
                ax[1,0].plot([self.yarr[ij],self.yarr[ij+1]],[self.zarr[ij],self.zarr[ij+1]])
                # ax2.plot([self.xarr[ij],self.xarr[ij+1]],[self.yarr[ij],self.yarr[ij+1]],[self.zarr[ij],self.zarr[ij+1]],color='tab:blue')
                # ax2.plot([self.xarr[ij],self.xarr[ij+1]],[self.yarr2[ij],self.yarr2[ij+1]],[self.zarr2[ij],self.zarr2[ij+1]],color='tab:blue')
                
                # ax2.plot([xarr2[i],xarr2[ij+1]],[yarr2[i],yarr2[ij+1]],[zarr2[i],zarr2[i+1]],color='tab:blue')
            ax[1,1].plot(uind)
            ax[1,1].grid()
            # surf=ax.plot_surface(mesh.X_nodes,mesh.Y_nodes,IC,cmap=cm.coolwarm)
            # ax2.scatter(Xp[0],Xp[1],Xp[2])
            print(np.sum(uind))

        # print(np.sum(uind))
        return [sum(cuind),sum(cvind),sum(cwind)],[sum(uind),sum(vind),sum(wind)]


    def Make_ind_matrix(self, controlpoints, rings, plot=True):

        n_cp = len(controlpoints)
        Au = np.zeros((n_cp, n_cp))
        Av = np.zeros((n_cp, n_cp))
        Aw = np.zeros((n_cp, n_cp))

        for target_idx, Xp in enumerate(controlpoints):
            for source_idx, filaments in enumerate(rings):
                u_total = 0.0
                v_total = 0.0
                w_total = 0.0

                for X1, X2 in filaments:
                    _, uvwind = self.biot_savart(X1, X2, Xp, 1.0)
                    u_total += uvwind[0]
                    v_total += uvwind[1]
                    w_total += uvwind[2]

                Au[target_idx, source_idx] = u_total
                Av[target_idx, source_idx] = v_total
                Aw[target_idx, source_idx] = w_total

        if plot == True:
            fig = plt.figure()
            ax = fig.subplots(1, 3)
            ax[0].imshow(Au)
            ax[0].set_title('U')
            ax[1].imshow(Av)
            ax[1].set_title('V')
            ax[2].imshow(Aw)
            ax[2].set_title('W')
            plt.show()

        return Au, Av, Aw



    def Lifting_line(self, resolution=100,a_ind_wake=0.,tend=5, tolerance=1e-6, max_iterations=1000, spacing='linear', use_prandtl=True, track_convergence=False, plot_geometry=False):
        cl_interp = interp1d(self.AoA, self.cl, kind='linear', fill_value='extrapolate')
        cd_interp = interp1d(self.AoA, self.cd, kind='linear', fill_value='extrapolate')
        self.resolution=resolution
        self.omega = (2 * np.pi * self.rpm) / 60
        A_disk = np.pi * self.radius**2

        # Generate normalized radial stations (0 to 1)
        if spacing == 'cosine': # cosine
            theta = np.linspace(0, np.pi, resolution + 1)
            r_normalized_temp = 0.5 * (1 - np.cos(theta))
            r_stations_norm = self.blade_start_fraction + (1 - self.blade_start_fraction) * r_normalized_temp
        else:  # linear
            r_stations_norm = np.linspace(self.blade_start_fraction, 1, resolution + 1)

        r_stations_norm = np.insert(r_stations_norm, 0, 0)
        self.r_stations_abs = r_stations_norm * self.radius
        self.dr =  np.diff(self.r_stations_abs)

        def blade_props(r_norm):
            if r_norm >= self.blade_start_fraction:
                twist = -50 * r_norm + 35 + self.collective_blade_pitch + self.collective_blade_pitch_location * 50 - 35
                chord = 0.18 - 0.06 * r_norm
            else:
                twist, chord = 0.0, 0.0
            return twist, chord

        # Node (panel-edge) properties: used to lay down the bound vortex and trailing legs.
        node_twist = np.array([blade_props(rn)[0] for rn in r_stations_norm])
        node_chord_norm = np.array([blade_props(rn)[1] for rn in r_stations_norm])

        r_control_abs = 0.5 * (self.r_stations_abs[:-1] + self.r_stations_abs[1:])
        r_control_norm = r_control_abs / self.radius
        twist_stations = np.array([blade_props(rn)[0] for rn in r_control_norm])
        chord_norm_stations = np.array([blade_props(rn)[1] for rn in r_control_norm])

        cp_chord_frac = getattr(self, 'cp_chord_frac', 0.25)      # chordwise cp position (co-located with bound vortex)
        trail_chord_frac = getattr(self, 'trail_chord_frac', 1.25)  # where the on-blade trailing legs end

        def rot_yz(vec,angle):
            c=np.cos(angle)
            s=np.sin(angle)
            return np.array([vec[0],vec[1]*c-vec[2]*s,vec[1]*s+vec[2]*c],dtype=float)

        def make_the_rotor():
            filament_rings=[]
            controlpoints=[]
            panels=[]
            theta_array=self.omega * self.tlst
            wake_pitch=self.U_inf/(self.omega+1e-16) 

            for blade in range(self.n_blades):
                angle_rotation=2*np.pi/self.n_blades*blade
                for i in range(len(self.r_stations_abs)-1):
                    r_in=self.r_stations_abs[i]
                    r_out=self.r_stations_abs[i+1]

                    # Node (panel-edge) chord/TE geometry for the bound vortex and trailing legs.
                    chord_in = node_chord_norm[i] * self.radius
                    twist_in_rad = np.radians(node_twist[i])
                    x_te_in = chord_in * np.sin(-twist_in_rad)
                    z_te_in = -chord_in * np.cos(twist_in_rad)

                    chord_out = node_chord_norm[i+1] * self.radius
                    twist_out_rad = np.radians(node_twist[i+1])
                    x_te_out = chord_out * np.sin(-twist_out_rad)
                    z_te_out = -chord_out * np.cos(twist_out_rad)

                    # Control point at the panel MIDSPAN, co-located chordwise with the bound
                    # vortex (c/4). This samples the induced velocity on the lifting line, so the
                    # induction magnitude matches momentum theory / BEM.
                    r_mid = r_control_abs[i]
                    chord_mid = chord_norm_stations[i] * self.radius
                    twist_mid_rad = np.radians(twist_stations[i])
                    x_te_mid = chord_mid * np.sin(-twist_mid_rad)
                    z_te_mid = -chord_mid * np.cos(twist_mid_rad)
                    cp = rot_yz([cp_chord_frac * x_te_mid, r_mid, cp_chord_frac * z_te_mid], angle_rotation)
                    controlpoints.append(cp)
                    panels.append(i)

                    # Quarter-chord points (LE is the chord origin, so c/4 = 0.25 * TE offset).
                    quarter_in = [0.25 * x_te_in, r_in, 0.25 * z_te_in]
                    quarter_out = [0.25 * x_te_out, r_out, 0.25 * z_te_out]

                    # Trailing legs run from the c/4 bound vortex to trail_chord_frac * chord
                    # (a quarter chord behind the TE by default); the wake convects from there.
                    trail_in = [trail_chord_frac * x_te_in, r_in, trail_chord_frac * z_te_in]
                    trail_out = [trail_chord_frac * x_te_out, r_out, trail_chord_frac * z_te_out]

                    filaments=[]
                    # Bound vortex at the quarter chord.
                    filaments.append((rot_yz(quarter_in, angle_rotation), rot_yz(quarter_out, angle_rotation)))

                    # Inboard on-blade trailing leg: c/4-behind-TE back to the quarter-chord bound vortex.
                    filaments.append((rot_yz(trail_in, angle_rotation), rot_yz(quarter_in, angle_rotation)))

                    for j in range(len(theta_array) - 1):
                        xt = filaments[-1][0][0]; yt = filaments[-1][0][1]; zt = filaments[-1][0][2]
                        dy = (np.cos(-theta_array[j+1]) - np.cos(-theta_array[j])) * r_in
                        dz = (np.sin(-theta_array[j+1]) - np.sin(-theta_array[j])) * r_in
                        dx = (theta_array[j+1] - theta_array[j]) / (np.pi / (self.radius * self.J))*(1+a_ind_wake)
                        dx, dy, dz = rot_yz([dx, dy, dz], angle_rotation)
                        filaments.append((np.array([xt+dx, yt+dy, zt+dz]), np.array([xt, yt, zt])))


                    # Outboard on-blade trailing leg: quarter-chord bound vortex out to c/4-behind-TE.
                    filaments.append((rot_yz(quarter_out, angle_rotation), rot_yz(trail_out, angle_rotation)))

                    for j in range(len(theta_array) - 1):
                        xt = filaments[-1][1][0]; yt = filaments[-1][1][1]; zt = filaments[-1][1][2]
                        dy = (np.cos(-theta_array[j+1]) - np.cos(-theta_array[j])) * r_out
                        dz = (np.sin(-theta_array[j+1]) - np.sin(-theta_array[j])) * r_out
                        dx = (theta_array[j+1] - theta_array[j]) / (np.pi / (self.radius * self.J))*(1+a_ind_wake)
                        dx, dy, dz = rot_yz([dx, dy, dz], angle_rotation)
                        filaments.append((np.array([xt, yt, zt]), np.array([xt+dx, yt+dy, zt+dz])))



                    # old=rot_yz([0,0,r_out],angle_rotation)
                    # for j in range(len(theta_array)-1):
                    #     th=theta_array[j+1]
                    #     t_cur = self.tlst[j+1]
                    #     new=rot_yz([t_cur*self.U_inf*(1+a_ind_wake), r_out*np.sin(-th), r_out*np.cos(-th)],angle_rotation)
                    #     filaments.append((old,new))
                    #     old=new

                    # helix=[]
                    # for j in range(len(theta_array)):
                    #     th=theta_array[j]
                    #     helix.append(rot_yz([wake_pitch*th, r_in*np.sin(-th), r_in*np.cos(-th)],angle_rotation))
                    # for j in range(len(helix)-1,0,-1):
                    #     filaments.append((helix[j],helix[j-1]))

                    filament_rings.append(filaments)

            return controlpoints,filament_rings,panels

        print(f"making {self.n_blades} blade lifting line geometry")
        controlpoints,rings,panels=make_the_rotor()
        n_cp=len(controlpoints)

        # Debug plot: show control points and filament rings (3D)
        if plot_geometry:
            try:
                fig3 = plt.figure()
                ax3 = fig3.add_subplot(projection='3d')
                ax3.set_proj_type('ortho')
                ax3.view_init(elev=90, azim=0)
                cps = np.array(controlpoints)
                if cps.size:
                    ax3.scatter(cps[:, 0], cps[:, 1], cps[:, 2], c='r', marker='', label='control points')

                # Draw filament segments
                for ring in rings:
                    for P1, P2 in ring:
                        xs = [P1[0], P2[0]]
                        ys = [P1[1], P2[1]]
                        zs = [P1[2], P2[2]]
                        ax3.plot(xs, ys, zs, c='b', marker='.')  # smaller blue dots for filaments

                ax3.set_title('Control points and filament rings')
                ax3.set_xlabel('X')
                ax3.set_ylabel('Y')
                ax3.set_zlabel('Z')
                ax3.legend()
                ax3.set_aspect('equalyz')
                plt.show()
            except Exception:
                pass

        

        print("making induced velocity matrix, this is the slow bit")
        MatrixU, MatrixV, MatrixW = self.Make_ind_matrix(controlpoints, rings, plot=False)

        GammaNew=np.zeros(n_cp)
        Gamma=np.zeros(n_cp)
        error=10
        ConvWeight=0.1
        converged_iter=max_iterations

        self.convergence_history = {'error': [], 'iteration': []} if track_convergence else None

        for kiter in range(max_iterations):
            Gamma=GammaNew.copy()

            u_ind=-MatrixU @ Gamma
            v_ind=-MatrixV @ Gamma
            w_ind=-MatrixW @ Gamma

            a_temp = np.zeros(n_cp)
            alpha_temp = np.zeros(n_cp)
            phi_temp = np.zeros(n_cp)
            aline_temp = np.zeros(n_cp)
            Fnorm_temp = np.zeros(n_cp)
            Ftan_temp = np.zeros(n_cp)

            for icp in range(n_cp):
                panel_i=panels[icp]
                r_norm=r_control_norm[panel_i]
                r_abs=r_control_abs[panel_i]

                if r_norm <= self.blade_start_fraction:
                    GammaNew[icp]=0
                    continue

                Xp=controlpoints[icp]
                radialposition=max(np.linalg.norm(Xp),1e-12)
                vrot=np.cross([-self.omega,0,0],Xp)
                vel1=np.array([self.U_inf+u_ind[icp]+vrot[0],v_ind[icp]+vrot[1],w_ind[icp]+vrot[2]],dtype=float)
                azimdir=np.cross(np.array([-1/radialposition,0,0]),Xp)
                vazim=np.dot(azimdir,vel1)
                vaxial=vel1[0]

                chord_abs=chord_norm_stations[panel_i]*self.radius
                theta_deg=twist_stations[panel_i]
                theta_rad=np.deg2rad(theta_deg)

                V_effective=np.sqrt(vaxial**2+vazim**2)
                phi=np.arctan2(vaxial,vazim)
                alpha=np.rad2deg(theta_rad-phi)
                alpha=np.clip(alpha, self.AoA[0], self.AoA[-1])
                cl=float(cl_interp(alpha))

                
                GammaNew[icp]=0.5*V_effective*chord_abs*cl

                a_temp[icp] = -1.*(-u_ind[icp] + vrot[0]) / (self.U_inf + 1e-12)
                alpha_temp[icp] = np.rad2deg(theta_rad - phi)
                phi_temp[icp] = np.rad2deg(phi)
                aline_temp[icp] = -1.*(vazim/(radialposition*self.omega) - 1)
                Fnorm_temp[icp] = cl * 0.5 * self.rho * V_effective**2 * chord_abs * np.cos(phi) - 0.5 * self.rho * V_effective**2 * chord_abs * cd_interp(alpha) * np.sin(phi)
                Ftan_temp[icp] = cl * 0.5 * self.rho * V_effective**2 * chord_abs * np.sin(phi) + 0.5 * self.rho * V_effective**2 * chord_abs * cd_interp(alpha) * np.cos(phi)


            refererror=max(np.max(np.abs(GammaNew)),0.001)
            error=np.max(np.abs(GammaNew-Gamma))/refererror
            if track_convergence:
                self.convergence_history['error'].append(error)
                self.convergence_history['iteration'].append(kiter)
            if kiter%25==0:
                print("ll iter",kiter,"error",error)
            if error<tolerance:
                converged_iter=kiter+1
                print("Lifting line converged after",converged_iter,"iterations")
                break

            GammaNew=(1-ConvWeight)*Gamma + ConvWeight*GammaNew

        if error>=tolerance:
            print("Warning: Lifting line did not converge but it did stop")

        return a_temp, aline_temp, Fnorm_temp, Ftan_temp, GammaNew, converged_iter, self.convergence_history, r_control_abs, alpha_temp, phi_temp

        

    # ==================================================================
    #  FREE WAKE VORTEX MODEL
    #  The wake is not prescribed: every wake node is convected by the
    #  freestream plus the velocity induced by the bound circulation AND
    #  by the wake itself, so the wake geometry (pitch + radial
    #  contraction/expansion) deforms until it is self-consistent.
    # ==================================================================

    @staticmethod
    def _rot_yz(vec, angle):
        # Rotation about the rotor (x) axis, matching the convention used by
        # rot_yz() inside Lifting_line().
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array([vec[0], vec[1] * c - vec[2] * s, vec[1] * s + vec[2] * c], dtype=float)

    def _core_at_radius_vec(self, r_abs):
        # Vectorised version of _core_at_radius(): graded Rankine core radius.
        r_abs = np.asarray(r_abs, dtype=float)
        if self.vortex_core is not None:
            return np.full_like(r_abs, float(self.vortex_core))
        mu = r_abs / self.radius
        frac = np.clip((mu - self.blade_start_fraction) / (1.0 - self.blade_start_fraction), 0.0, 1.0)
        return self.vortex_core_root + (self.vortex_core_tip - self.vortex_core_root) * frac

    def _bs_core(self, P1, P2, rc, Xp):
        # Vectorised Biot-Savart. Returns the UNIT (gamma = 1) induced velocity
        # at every evaluation point Xp[C,3] from every straight filament segment
        # (P1[S,3] -> P2[S,3]) with Rankine core rc[S]. Result shape (C, S, 3).
        # This is the exact same regularised kernel as the scalar biot_savart().
        eps = 1e-6
        Xp = np.asarray(Xp, dtype=float)[:, None, :]      # (C,1,3)
        P1 = np.asarray(P1, dtype=float)[None, :, :]       # (1,S,3)
        P2 = np.asarray(P2, dtype=float)[None, :, :]       # (1,S,3)
        rc = np.asarray(rc, dtype=float)[None, :]          # (1,S)

        R1v = Xp - P1                                      # (C,S,3)
        R2v = Xp - P2
        R1 = np.maximum(np.sqrt(np.sum(R1v * R1v, axis=-1)), eps)   # (C,S)
        R2 = np.maximum(np.sqrt(np.sum(R2v * R2v, axis=-1)), eps)

        cr = np.cross(R1v, R2v)                            # (C,S,3)
        cr_sq = np.sum(cr * cr, axis=-1)                   # (C,S)

        r0 = P2 - P1                                       # (1,S,3)
        r0_sq = np.sum(r0 * r0, axis=-1)                   # (1,S)
        cr_sq = cr_sq + (rc * rc) * r0_sq                  # core regularisation
        cr_sq = np.maximum(cr_sq, eps)

        R01 = np.sum(r0 * R1v, axis=-1)                    # (C,S)
        R02 = np.sum(r0 * R2v, axis=-1)
        K = (1.0 / (4.0 * np.pi * cr_sq)) * (R01 / R1 - R02 / R2)   # (C,S)
        return K[:, :, None] * cr                          # (C,S,3)

    def _free_wake_blade_geometry(self, resolution, spacing='linear'):
        # Fixed (wake-independent) blade geometry: radial stations, control
        # points, quarter-chord (bound) nodes Q and the age-0 wake anchor
        # points T0 (a quarter chord behind the TE), for every blade.
        if spacing == 'cosine':
            theta = np.linspace(0, np.pi, resolution + 1)
            r_norm_tmp = 0.5 * (1 - np.cos(theta))
            r_stations_norm = self.blade_start_fraction + (1 - self.blade_start_fraction) * r_norm_tmp
        else:
            r_stations_norm = np.linspace(self.blade_start_fraction, 1, resolution + 1)
        r_stations_norm = np.insert(r_stations_norm, 0, 0.0)

        self.r_stations_abs = r_stations_norm * self.radius
        self.dr = np.diff(self.r_stations_abs)
        n_edges = len(r_stations_norm)
        n_panels = n_edges - 1

        def bp(r_norm):
            if r_norm >= self.blade_start_fraction:
                twist = -50 * r_norm + 35 + self.collective_blade_pitch + self.collective_blade_pitch_location * 50 - 35
                chord = 0.18 - 0.06 * r_norm
            else:
                twist, chord = 0.0, 0.0
            return twist, chord

        node_twist = np.array([bp(rn)[0] for rn in r_stations_norm])
        node_chord_norm = np.array([bp(rn)[1] for rn in r_stations_norm])

        r_control_abs = 0.5 * (self.r_stations_abs[:-1] + self.r_stations_abs[1:])
        r_control_norm = r_control_abs / self.radius
        twist_stations = np.array([bp(rn)[0] for rn in r_control_norm])
        chord_norm_stations = np.array([bp(rn)[1] for rn in r_control_norm])

        cp_chord_frac = getattr(self, 'cp_chord_frac', 0.25)
        trail_chord_frac = getattr(self, 'trail_chord_frac', 1.25)

        Q = np.zeros((self.n_blades, n_edges, 3))    # quarter-chord (bound) nodes
        T0 = np.zeros((self.n_blades, n_edges, 3))   # age-0 wake anchor points
        for b in range(self.n_blades):
            ang = 2 * np.pi / self.n_blades * b
            for j in range(n_edges):
                chord = node_chord_norm[j] * self.radius
                tw = np.radians(node_twist[j])
                x_te = chord * np.sin(-tw)
                z_te = -chord * np.cos(tw)
                r_j = self.r_stations_abs[j]
                Q[b, j] = self._rot_yz([0.25 * x_te, r_j, 0.25 * z_te], ang)
                T0[b, j] = self._rot_yz([trail_chord_frac * x_te, r_j, trail_chord_frac * z_te], ang)

        controlpoints = []
        panels = []
        for b in range(self.n_blades):
            ang = 2 * np.pi / self.n_blades * b
            for i in range(n_panels):
                chord_mid = chord_norm_stations[i] * self.radius
                tw_mid = np.radians(twist_stations[i])
                x_te_mid = chord_mid * np.sin(-tw_mid)
                z_te_mid = -chord_mid * np.cos(tw_mid)
                cp = self._rot_yz([cp_chord_frac * x_te_mid, r_control_abs[i], cp_chord_frac * z_te_mid], ang)
                controlpoints.append(cp)
                panels.append(i)

        return dict(n_edges=n_edges, n_panels=n_panels, Q=Q, T0=T0,
                    controlpoints=np.array(controlpoints), panels=panels,
                    r_control_abs=r_control_abs, r_control_norm=r_control_norm,
                    twist_stations=twist_stations, chord_norm_stations=chord_norm_stations)

    def _initial_wake(self, geom, a_ind_wake):
        # Starting guess: prescribed constant-pitch helix at fixed shedding
        # radius (identical in spirit to Lifting_line's frozen wake). The
        # relaxation deforms it from here.
        T0 = geom['T0']
        n_edges = geom['n_edges']
        n_age = len(self.tlst)
        dt = self.tlst[1] - self.tlst[0]
        W = np.zeros((self.n_blades, n_edges, n_age, 3))
        for b in range(self.n_blades):
            for j in range(n_edges):
                x0, y0, z0 = T0[b, j]
                r0 = np.hypot(y0, z0)
                psi0 = np.arctan2(z0, y0)
                for k in range(n_age):
                    x = x0 + k * self.U_inf * (1 + a_ind_wake) * dt
                    psi = psi0 - k * self.omega * dt
                    W[b, j, k] = [x, r0 * np.cos(psi), r0 * np.sin(psi)]
        return W

    def _assemble_segments(self, W, geom):
        # Build the full straight-filament list (bound + on-blade trailing legs
        # + free wake) from the current wake node grid W. Every segment is
        # tagged with the global panel index (0..n_cp-1) whose circulation it
        # carries, so a single Gamma vector drives the whole vortex system.
        Q = geom['Q']
        n_panels = geom['n_panels']
        n_age = W.shape[2]
        P1, P2, pan = [], [], []
        for b in range(self.n_blades):
            for i in range(n_panels):
                p = b * n_panels + i
                # Bound vortex (quarter chord), edge i -> i+1
                P1.append(Q[b, i]);       P2.append(Q[b, i + 1]);   pan.append(p)
                # Inboard on-blade trailing leg: age-0 wake point -> quarter chord
                P1.append(W[b, i, 0]);    P2.append(Q[b, i]);       pan.append(p)
                # Inboard trailing wake at edge i (oriented downstream -> blade)
                for k in range(n_age - 1):
                    P1.append(W[b, i, k + 1]); P2.append(W[b, i, k]); pan.append(p)
                # Outboard on-blade trailing leg: quarter chord -> age-0 wake point
                P1.append(Q[b, i + 1]);   P2.append(W[b, i + 1, 0]); pan.append(p)
                # Outboard trailing wake at edge i+1 (oriented blade -> downstream)
                for k in range(n_age - 1):
                    P1.append(W[b, i + 1, k]); P2.append(W[b, i + 1, k + 1]); pan.append(p)
        P1 = np.array(P1)
        P2 = np.array(P2)
        pan = np.array(pan)
        r_seg = 0.5 * (np.hypot(P1[:, 1], P1[:, 2]) + np.hypot(P2[:, 1], P2[:, 2]))
        rc = self._core_at_radius_vec(r_seg)
        return P1, P2, rc, pan

    def _influence_matrices(self, P1, P2, rc, pan, controlpoints, n_cp):
        # Assemble the (n_cp x n_cp) induced-velocity matrices at the control
        # points, summing each panel's filaments into that panel's column.
        V = self._bs_core(P1, P2, rc, controlpoints)      # (C, S, 3)
        S = P1.shape[0]
        onehot = np.zeros((S, n_cp))
        onehot[np.arange(S), pan] = 1.0
        Au = V[:, :, 0] @ onehot
        Av = V[:, :, 1] @ onehot
        Aw = V[:, :, 2] @ onehot
        return Au, Av, Aw

    def _node_induction(self, P1, P2, rc, gamma_seg, Xp, chunk=256, wake_core=None):
        # Gamma-weighted induced velocity at every wake node Xp[C,3]. Chunked
        # over evaluation points to keep the (chunk x S x 3) temporary small.
        # An optional larger wake_core desingularises the wake-on-wake
        # interaction (damps the grid-scale far-wake jitter) without changing
        # the blade loads, which use the finer graded core.
        if wake_core is not None:
            rc = np.maximum(rc, wake_core)
        C = Xp.shape[0]
        V = np.zeros((C, 3))
        for s in range(0, C, chunk):
            e = min(s + chunk, C)
            Vc = self._bs_core(P1, P2, rc, Xp[s:e])       # (nc, S, 3)
            V[s:e, 0] = Vc[:, :, 0] @ gamma_seg
            V[s:e, 1] = Vc[:, :, 1] @ gamma_seg
            V[s:e, 2] = Vc[:, :, 2] @ gamma_seg
        return V

    def _solve_gamma(self, MatrixU, MatrixV, MatrixW, controlpoints, panels,
                     r_control_norm, r_control_abs, chord_norm_stations, twist_stations,
                     cl_interp, cd_interp, Gamma_init, tolerance, max_iterations, ConvWeight=0.1):
        # Fixed-point circulation solve for a GIVEN wake geometry. Physics is
        # identical to the inner loop of Lifting_line().
        n_cp = len(controlpoints)
        GammaNew = Gamma_init.copy()
        error = 10.0
        converged_iter = max_iterations
        a_temp = np.zeros(n_cp); alpha_temp = np.zeros(n_cp); phi_temp = np.zeros(n_cp)
        aline_temp = np.zeros(n_cp); Fnorm_temp = np.zeros(n_cp); Ftan_temp = np.zeros(n_cp)

        for kiter in range(max_iterations):
            Gamma = GammaNew.copy()
            u_ind = -MatrixU @ Gamma
            v_ind = -MatrixV @ Gamma
            w_ind = -MatrixW @ Gamma

            for icp in range(n_cp):
                panel_i = panels[icp]
                r_norm = r_control_norm[panel_i]
                r_abs = r_control_abs[panel_i]
                if r_norm <= self.blade_start_fraction:
                    GammaNew[icp] = 0
                    continue

                Xp = controlpoints[icp]
                radialposition = max(np.linalg.norm(Xp), 1e-12)
                vrot = np.cross([-self.omega, 0, 0], Xp)
                vel1 = np.array([self.U_inf + u_ind[icp] + vrot[0], v_ind[icp] + vrot[1], w_ind[icp] + vrot[2]], dtype=float)
                azimdir = np.cross(np.array([-1 / radialposition, 0, 0]), Xp)
                vazim = np.dot(azimdir, vel1)
                vaxial = vel1[0]

                chord_abs = chord_norm_stations[panel_i] * self.radius
                theta_rad = np.deg2rad(twist_stations[panel_i])

                V_effective = np.sqrt(vaxial**2 + vazim**2)
                phi = np.arctan2(vaxial, vazim)
                alpha = np.rad2deg(theta_rad - phi)
                alpha = np.clip(alpha, self.AoA[0], self.AoA[-1])
                cl = float(cl_interp(alpha))
                cd = float(cd_interp(alpha))

                GammaNew[icp] = 0.5 * V_effective * chord_abs * cl

                a_temp[icp] = -1. * (-u_ind[icp] + vrot[0]) / (self.U_inf + 1e-12)
                alpha_temp[icp] = np.rad2deg(theta_rad - phi)
                phi_temp[icp] = np.rad2deg(phi)
                aline_temp[icp] = -1. * (vazim / (radialposition * self.omega) - 1)
                Fnorm_temp[icp] = cl * 0.5 * self.rho * V_effective**2 * chord_abs * np.cos(phi) - 0.5 * self.rho * V_effective**2 * chord_abs * cd * np.sin(phi)
                Ftan_temp[icp] = cl * 0.5 * self.rho * V_effective**2 * chord_abs * np.sin(phi) + 0.5 * self.rho * V_effective**2 * chord_abs * cd * np.cos(phi)

            refererror = max(np.max(np.abs(GammaNew)), 0.001)
            error = np.max(np.abs(GammaNew - Gamma)) / refererror
            if error < tolerance:
                converged_iter = kiter + 1
                break
            GammaNew = (1 - ConvWeight) * Gamma + ConvWeight * GammaNew

        return GammaNew, a_temp, aline_temp, Fnorm_temp, Ftan_temp, alpha_temp, phi_temp, converged_iter

    def Lifting_line_freewake(self, resolution=15, a_ind_wake=0.2, spacing='linear',
                              wake_iterations=30, wake_relax=0.3, wake_tol=1e-3,
                              gamma_tol=1e-6, gamma_max_iter=1000, axial_floor_frac=0.3,
                              wake_core=0.10, verbose=True):
        """Free (deforming) wake lifting-line solver.

        Outer relaxation:
          1. build the vortex filaments from the current wake geometry,
          2. solve the bound circulation Gamma for that geometry,
          3. evaluate the velocity induced by the whole vortex system (bound
             + wake) at every wake node,
          4. re-integrate each trailing line from the blade, convecting nodes
             with U_inf + V_induced (axial pitch + radial contraction free;
             the -Omega*dt azimuth step is the shedding rotation that forms
             the helix),
          5. under-relax and repeat until the wake stops moving.

        wake_core enlarges the Rankine core used only for the wake-on-wake
        induction, which damps the grid-scale far-wake jitter without changing
        the blade loads (those keep the finer graded core).

        Requires self.tlst to be set (defines wake length and step), as in
        Lifting_line().
        """
        cl_interp = interp1d(self.AoA, self.cl, kind='linear', fill_value='extrapolate')
        cd_interp = interp1d(self.AoA, self.cd, kind='linear', fill_value='extrapolate')
        self.resolution = resolution
        self.omega = (2 * np.pi * self.rpm) / 60
        dt = self.tlst[1] - self.tlst[0]
        n_age = len(self.tlst)
        r_floor = 1e-3 * self.radius

        geom = self._free_wake_blade_geometry(resolution, spacing)
        controlpoints = geom['controlpoints']
        panels = geom['panels']
        r_control_abs = geom['r_control_abs']
        r_control_norm = geom['r_control_norm']
        twist_stations = geom['twist_stations']
        chord_norm_stations = geom['chord_norm_stations']
        n_panels = geom['n_panels']
        n_edges = geom['n_edges']
        n_cp = len(controlpoints)

        print(f"making {self.n_blades} blade free-wake lifting line geometry "
              f"({n_cp} control points, {n_age} wake nodes/line)")

        W0 = self._initial_wake(geom, a_ind_wake)
        W = W0.copy()
        Gamma = np.zeros(n_cp)

        wake_move_hist = []
        a_temp = aline_temp = Fnorm_temp = Ftan_temp = alpha_temp = phi_temp = None
        used_iter = wake_iterations

        for wit in range(wake_iterations):
            # (1) filaments from the current wake geometry
            P1, P2, rc, pan = self._assemble_segments(W, geom)

            # (2) influence matrices + circulation solve
            MatrixU, MatrixV, MatrixW = self._influence_matrices(P1, P2, rc, pan, controlpoints, n_cp)
            (Gamma, a_temp, aline_temp, Fnorm_temp, Ftan_temp,
             alpha_temp, phi_temp, giter) = self._solve_gamma(
                MatrixU, MatrixV, MatrixW, controlpoints, panels,
                r_control_norm, r_control_abs, chord_norm_stations, twist_stations,
                cl_interp, cd_interp, Gamma, gamma_tol, gamma_max_iter)

            # (3) velocity induced by the whole vortex system at every wake node.
            #     Same sign convention as the blade solve (-Matrix @ Gamma).
            gamma_seg = Gamma[pan]
            Vind = -self._node_induction(P1, P2, rc, gamma_seg, W.reshape(-1, 3), wake_core=wake_core)
            Vind = Vind.reshape(self.n_blades, n_edges, n_age, 3)

            # (4) re-integrate each trailing line from its anchored age-0 node
            Wnew = W.copy()
            for b in range(self.n_blades):
                for j in range(n_edges):
                    for k in range(n_age - 1):
                        x, y, z = Wnew[b, j, k]
                        r = max(np.hypot(y, z), r_floor)
                        ct = y / r
                        st = z / r
                        vx = self.U_inf + Vind[b, j, k, 0]
                        vr = Vind[b, j, k, 1] * ct + Vind[b, j, k, 2] * st
                        vt = -Vind[b, j, k, 1] * st + Vind[b, j, k, 2] * ct
                        vx = max(vx, axial_floor_frac * self.U_inf)   # keep wake convecting downstream
                        x_new = x + vx * dt
                        r_new = max(r + vr * dt, r_floor)
                        psi = np.arctan2(z, y)
                        psi_new = psi + (vt / r) * dt - self.omega * dt
                        Wnew[b, j, k + 1] = [x_new, r_new * np.cos(psi_new), r_new * np.sin(psi_new)]

            # (5) under-relax and check convergence
            dmax = float(np.max(np.linalg.norm(Wnew - W, axis=-1)))
            W = (1 - wake_relax) * W + wake_relax * Wnew
            wake_move_hist.append(dmax)
            if verbose:
                print(f"[free-wake] iter {wit + 1}/{wake_iterations}  "
                      f"max node move = {dmax:.4e} m  (gamma iters {giter})")
            if dmax < wake_tol * self.radius:
                used_iter = wit + 1
                if verbose:
                    print(f"[free-wake] wake geometry converged after {used_iter} iterations")
                break

        self.free_wake_W = W
        self.free_wake_W0 = W0
        convergence_history = {'error': wake_move_hist, 'iteration': list(range(len(wake_move_hist)))}
        return (a_temp, aline_temp, Fnorm_temp, Ftan_temp, Gamma, used_iter,
                convergence_history, r_control_abs, alpha_temp, phi_temp, W0, W)

    def blade_element(self, resolution=100, tolerance=1e-6, max_iterations=1000, spacing='linear', use_prandtl=True, track_convergence=False):
        
        cl_interp = interp1d(self.AoA, self.cl, kind='linear', fill_value='extrapolate')
        cd_interp = interp1d(self.AoA, self.cd, kind='linear', fill_value='extrapolate')
        
        omega = (2 * np.pi * self.rpm) / 60
        
        # Generate normalized radial stations (0 to 1)
        if spacing == 'cosine': # cosine
            theta = np.linspace(0, np.pi, resolution + 1)
            r_normalized_temp = 0.5 * (1 - np.cos(theta))
            r_stations_norm = self.blade_start_fraction + (1 - self.blade_start_fraction) * r_normalized_temp
        else:  # linear
            r_stations_norm = np.linspace(self.blade_start_fraction, 1, resolution + 1)
        
        # Regenerate normalized blade properties at new radial stations
        twist_stations = []
        chord_norm_stations = []
        r_stations_norm = np.insert(r_stations_norm, 0, 2*r_stations_norm[0]-r_stations_norm[1])
        r_stations_norm = np.insert(r_stations_norm, 0, 0)
        
        for r_norm in r_stations_norm:
            if r_norm > self.blade_start_fraction:
                twist_stations.append(-50 * r_norm + 35 + self.collective_blade_pitch + self.collective_blade_pitch_location * 50 - 35 )
                chord_norm_stations.append(0.18 - 0.06 * r_norm)
            else:
                twist_stations.append(0)
                chord_norm_stations.append(0)
        
        # Calculate self.dr in absolute units
        r_stations_abs = r_stations_norm * self.radius
        self.dr =  np.diff(r_stations_abs)
        #self.dr = np.append(self.dr, self.dr[-1])
        #self.dr = np.delete(self.dr, 0)
        
        A_disk = np.pi * self.radius**2
        
        # Initialize tracking lists (all use NORMALIZED radius 0-1)
        self.r_R_list = []
        self.a_list = []
        self.a_prime_list = []
        self.alpha_list = []
        self.phi_list = []
        self.cl_list = []
        self.cd_list = []
        self.V_axial_list = []
        self.V_tangential_list = []
        self.V_effective_list = []
        self.lift_list = []
        self.drag_list = []
        self.F_axial_list = []
        self.F_azimuth_list = []
        self.circulation_list = []
        self.F_prandtl_list = []
        self.dCT_dr_list = []
        self.CT_conv_list = []
        self.CT_conv_ind=[]
        self.dCQ_dr_list = []
        self.dCP_dr_list = []
        self.iterations_list = []
        
        # Convergence tracking
        if track_convergence:
            self.convergence_history = {'CT': [], 'CQ': [], 'iteration': []}
        
        dT_total = 0
        dQ_total = 0
        
        for i, r_norm in enumerate(r_stations_norm):

            a = 0.3
            a_prime = 0.
            
            # Get absolute radius for calculations
            r_abs = r_norm * self.radius
            

            if r_norm <= self.blade_start_fraction or (use_prandtl and r_norm >= 1.0):
                self.r_R_list.append(r_norm)
                self.a_list.append(0)
                self.a_prime_list.append(0)
                self.alpha_list.append(0)
                self.phi_list.append(0)
                self.cl_list.append(0)
                self.cd_list.append(0)
                self.V_axial_list.append(0)
                self.V_tangential_list.append(0)
                self.V_effective_list.append(0)
                self.lift_list.append(0)
                self.drag_list.append(0)
                self.F_axial_list.append(0)
                self.F_azimuth_list.append(0)
                self.circulation_list.append(0)
                self.F_prandtl_list.append(0)
                self.dCT_dr_list.append(0)
                self.dCQ_dr_list.append(0)
                self.dCP_dr_list.append(0)
                self.iterations_list.append(0)
                continue
            
            iter_count = 0
            converged = False
            for iteration in range(max_iterations):
                iter_count += 1
                
                # Velocities (absolute)
                V_axial = self.U_inf * (1 + a)
                V_tangential = omega * r_abs * (1 - a_prime)
                V_effective = np.sqrt(V_axial**2 + V_tangential**2)
                
                # Get blade properties (convert normalized chord to absolute)
                chord_abs = chord_norm_stations[i] * self.radius
                theta_deg = twist_stations[i]
                theta_rad = np.deg2rad(theta_deg)
                
                # Flow angles
                phi = np.arctan2(V_axial, V_tangential)
                alpha = np.rad2deg(theta_rad - phi)
                alpha = np.clip(alpha, self.AoA[0], self.AoA[-1])
                
                # Airfoil coefficients
                cl = float(cl_interp(alpha))
                cd = float(cd_interp(alpha))
                
                # Forces per unit length (absolute)
                lift = 0.5 * self.rho * V_effective**2 * chord_abs * cl
                self.drag = 0.5 * self.rho * V_effective**2 * chord_abs * cd
                
                # Apply Prandtl correction if enabled
                if use_prandtl:
                    F_prandtl = self._calculate_prandtl_factor(r_norm, a)
                else:
                    F_prandtl = 1.0
                # Floor F so the divisions below stay finite at the very tip (F -> 0 there).
                F_safe = max(F_prandtl, 1e-4)


                F_azimuth = lift * np.sin(phi) + self.drag * np.cos(phi)
                F_axial = lift * np.cos(phi) - self.drag * np.sin(phi)

                # Thrust coefficient from the real blade force.
                A_a = 2 * np.pi * r_abs * self.dr[i-1]
                C_T = (F_axial * self.n_blades * self.dr[i-1]) / (0.5 * self.rho * self.U_inf**2 * A_a)
                if i==20 and iter_count%2==0:
                    self.CT_conv_list.append(C_T)
                    self.CT_conv_ind.append(iter_count)


                a_calc = (-1/2) * (1 - np.sqrt(max(0, 1 + C_T / F_safe)))

                a_calc = np.clip(a_calc, 0, 0.95)
                a_prime_calc = (F_azimuth * self.n_blades) / (2 * self.rho * (2 * np.pi * r_abs) * self.U_inf * (1 + a_calc) * omega * r_abs * F_safe)
                
                # Check convergence
                if abs(a_calc - a) < tolerance and abs(a_prime_calc - a_prime) < tolerance:
                    converged = True
                    break
                
                # Relaxation of iterative variables
                a = 0.75 * a + 0.25 * a_calc
                a_prime = 0.75 * a_prime + 0.25 * a_prime_calc


            
            # Calculate circulation
            circulation = 0.5 * V_effective * chord_abs * cl
            
            # Accumulate totals
            dT_total += F_axial * self.dr[i-1] * self.n_blades
            dQ_total += F_azimuth * r_abs * self.dr[i-1] * self.n_blades
            
            # Differential coefficients
            dCT = C_T
            dCQ = (F_azimuth * self.n_blades * r_abs * self.dr[i-1]) / (0.5 * self.rho * self.U_inf**2 * A_a * self.radius)
            dCP = dCQ * omega * self.radius / self.U_inf
            
            # Store values (using NORMALIZED radius)
            self.r_R_list.append(r_norm)
            self.a_list.append(a_calc)
            self.a_prime_list.append(a_prime_calc)
            self.alpha_list.append(alpha)
            self.phi_list.append(np.rad2deg(phi))
            self.cl_list.append(cl)
            self.cd_list.append(cd)
            self.V_axial_list.append(V_axial)
            self.V_tangential_list.append(V_tangential)
            self.V_effective_list.append(V_effective)
            self.lift_list.append(lift)
            self.drag_list.append(self.drag)
            self.F_axial_list.append(F_axial)
            self.F_azimuth_list.append(F_azimuth)
            self.circulation_list.append(circulation)
            self.F_prandtl_list.append(F_prandtl)
            self.dCT_dr_list.append(dCT)
            self.dCQ_dr_list.append(dCQ)
            self.dCP_dr_list.append(dCP)
            self.iterations_list.append(iter_count)
            
            # Track convergence if requested
            if track_convergence:
                CT_current = dT_total / (0.5 * self.rho * self.U_inf**2 * A_disk)
                CQ_current = dQ_total / (0.5 * self.rho * self.U_inf**2 * A_disk * self.radius)
                self.convergence_history['CT'].append(CT_current)
                self.convergence_history['CQ'].append(CQ_current)
                self.convergence_history['iteration'].append(i)

            if converged == False:
                print(
                    f"Warning: Blade element method did not converge within the maximum iterations for station {r_abs} for spacing {spacing}.")
        
        # Total coefficients (propeller convention: rho * n^x * D^y)
        # C_T = T / (rho n^2 D^4), C_Q = Q / (rho n^2 D^5), C_P = P / (rho n^3 D^5)
        self.CT = dT_total / (self.rho * self.n_rps**2 * self.D**4)
        self.CQ = dQ_total / (self.rho * self.n_rps**2 * self.D**5)
        self.CP = (dQ_total * omega) / (self.rho * self.n_rps**3 * self.D**5)

        print(f"Total thrust coefficient from BEM:  C_T = {self.CT:.4f}")
        print(f"Total torque coefficient from BEM:  C_Q = {self.CQ:.4f}")
        print(f"Total power coefficient from BEM:   C_P = {self.CP:.4f}")



if __name__ == "__main__":

    # ------------------------------------------------------------------
    # Free Wake Vortex Model demo.
    # The wake is convected by the freestream plus the velocity induced by
    # the blades' bound circulation AND by the wake itself, so the wake
    # geometry (pitch + radial contraction) deforms until self-consistent.
    # ------------------------------------------------------------------
    bem = BEM(J=1.6, radius=0.7, n_blades=6, U_inf=60)

    tend = 0.3
    dt = 0.005
    bem.tlst = np.arange(0, tend, dt)

    output = bem.Lifting_line_freewake(
        resolution=20,
        a_ind_wake=0.2,        # initial helix pitch guess (deformed afterwards)
        spacing='linear',
        wake_iterations=100,
        wake_relax=0.3,
        wake_tol=1e-3,
        wake_core=0.10,        # desingularises wake-on-wake induction (stability)
    )

    (a_out, aline_out, Fnorm_out, Ftan_out, Gamma_out, wake_iter,
     wake_hist, r_control, alpha_out, phi_out, W0, W) = output

    # --- integrated coefficients (propeller convention) ---
    blade_count = bem.n_blades
    n_panels = len(r_control)
    T_LL = 0.0
    Q_LL = 0.0
    for p in range(1, n_panels):            # skip the root panel (no blade there)
        T_LL += Fnorm_out[p] * blade_count * bem.dr[p]
        Q_LL += Ftan_out[p] * blade_count * r_control[p] * bem.dr[p]

    C_T_LL = T_LL / (bem.rho * bem.n_rps**2 * bem.D**4)
    C_Q_LL = Q_LL / (bem.rho * bem.n_rps**2 * bem.D**5)
    C_P_LL = (Q_LL * bem.omega) / (bem.rho * bem.n_rps**3 * bem.D**5)
    print(f"Free-wake thrust coefficient: C_T = {C_T_LL:.4f}")
    print(f"Free-wake torque coefficient: C_Q = {C_Q_LL:.4f}")
    print(f"Free-wake power  coefficient: C_P = {C_P_LL:.4f}")

    # ------------------------------------------------------------------
    # 3D wake geometry: initial prescribed helix vs deformed free wake
    # ------------------------------------------------------------------
    n_edges = W.shape[1]
    fig_w = plt.figure(figsize=(11, 6))
    axw = fig_w.add_subplot(projection='3d')
    # all blades' free wake, faint, for context
    for b in range(bem.n_blades):
        for j in range(n_edges):
            axw.plot(W[b, j, :, 0], W[b, j, :, 1], W[b, j, :, 2], color='tab:blue', lw=0.3, alpha=0.25)
    # one blade highlighted: prescribed helix vs deformed free wake
    for j in range(n_edges):
        axw.plot(W0[0, j, :, 0], W0[0, j, :, 1], W0[0, j, :, 2], color='0.6', lw=0.9)
        axw.plot(W[0, j, :, 0], W[0, j, :, 1], W[0, j, :, 2], color='tab:red', lw=0.9)
    axw.plot([], [], [], color='0.6', label='initial prescribed helix (blade 0)')
    axw.plot([], [], [], color='tab:red', label='free wake (blade 0)')
    axw.plot([], [], [], color='tab:blue', alpha=0.4, label='free wake (other blades)')
    axw.set_title('Wake geometry: prescribed helix vs deformed free wake')
    axw.set_xlabel('x (axial)')
    axw.set_ylabel('y')
    axw.set_zlabel('z')
    axw.legend()
    try:
        axw.set_box_aspect((2, 1, 1))
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Radial deformation diagnostic (contraction / expansion of the wake)
    # ------------------------------------------------------------------
    fig_r, axr = plt.subplots(figsize=(7, 4))
    for j in range(n_edges):
        r0 = np.hypot(W0[0, j, :, 1], W0[0, j, :, 2])
        rf = np.hypot(W[0, j, :, 1], W[0, j, :, 2])
        axr.plot(W0[0, j, :, 0], r0, color='0.8', lw=0.6)
        axr.plot(W[0, j, :, 0], rf, color='tab:red', lw=0.6)
    axr.plot([], [], color='0.8', label='prescribed (fixed radius)')
    axr.plot([], [], color='tab:red', label='free wake')
    axr.set_xlabel('x (axial)')
    axr.set_ylabel('wake radius (m)')
    axr.set_title('Wake radius vs axial position (blade 0)')
    axr.legend()
    axr.grid(True)

    # ------------------------------------------------------------------
    # Spanwise loads / induction / circulation
    # ------------------------------------------------------------------
    # L = F_norm * cos(phi) + F_tan * sin(phi)
    Lift_dist = Fnorm_out * np.cos(np.deg2rad(phi_out)) + Ftan_out * np.sin(np.deg2rad(phi_out))
    # D = -F_norm * sin(phi) + F_tan * cos(phi)
    Drag_dist = -Fnorm_out * np.sin(np.deg2rad(phi_out)) + Ftan_out * np.cos(np.deg2rad(phi_out))

    station_count = len(r_control) + 1

    def plot_blade_overlay(ax, x_values, y_values, label_prefix='', style='-o'):
        x_values = np.asarray(x_values)
        y_values = np.asarray(y_values)
        x_masked = x_values[1:]
        if station_count > 0 and len(y_values) == blade_count * (station_count - 1):
            blade_series = [np.asarray(y_values[i * (station_count - 1):(i + 1) * (station_count - 1)])[1:] for i in range(blade_count)]
            if len(blade_series) > 0 and all(np.allclose(blade_series[0], series) for series in blade_series[1:]):
                ax.plot(x_masked, blade_series[0], style, label=f'{label_prefix} all blades (identical)')
                ax.text(0.02, 0.95, f'{blade_count} blades overlap', transform=ax.transAxes,
                        va='top', ha='left', fontsize=9)
            else:
                for blade_idx, series in enumerate(blade_series, start=1):
                    ax.plot(x_masked, series, style, label=f'{label_prefix} blade {blade_idx}')
        else:
            ax.plot(x_masked, y_values[1:], style, label=label_prefix)

    def finish_axis(ax, title, ylabel):
        ax.set_title(title)
        ax.set_xlabel('r (m)')
        ax.set_ylabel(ylabel)
        try:
            blade_root_r = bem.blade_start_fraction * bem.radius
            ax.axvline(blade_root_r, color='k', linestyle='--', linewidth=1, label='blade start')
        except Exception:
            pass
        ax.legend()
        ax.grid(True)

    fig, axs = plt.subplots(2, 3, figsize=(15, 8))

    # Circulation
    try:
        plot_blade_overlay(axs[0, 0], r_control, Gamma_out * bem.n_blades * np.pi * bem.omega / bem.U_inf**2, 'Gamma')
        finish_axis(axs[0, 0], 'Nondimensional Circulation vs radius', r'$\frac{\Gamma N_\text{blades} \pi \Omega}{U_\infty^2}$')
    except Exception:
        pass

    # Axial and azimuthal induction
    try:
        plot_blade_overlay(axs[0, 1], r_control, a_out, 'a')
        plot_blade_overlay(axs[0, 1], r_control, aline_out, "a'", style='-s')
        finish_axis(axs[0, 1], 'Induction factors', 'Induction factor')
    except Exception:
        pass

    # Angle of attack and flow angle
    try:
        plot_blade_overlay(axs[0, 2], r_control, alpha_out, 'AoA', style='-^')
        plot_blade_overlay(axs[0, 2], r_control, phi_out, 'Flow angle', style='-s')
        finish_axis(axs[0, 2], 'Angle of attack and Flow angle', r'AoA, $\phi$  (degrees)')
    except Exception:
        pass

    # Section forces
    try:
        plot_blade_overlay(axs[1, 0], r_control, Fnorm_out / (bem.rho * bem.n_rps**2 * bem.D**3), 'Fnorm')
        plot_blade_overlay(axs[1, 0], r_control, Ftan_out / (bem.rho * bem.n_rps**2 * bem.D**3), 'Ftan', style='-s')
        finish_axis(axs[1, 0], 'Section forces', r'$ C_F = \frac{1}{\rho n^2 D^3}\frac{dF}{dr}$')
    except Exception:
        pass

    # Wake-geometry convergence history
    try:
        if wake_hist is not None and len(wake_hist['error']) > 0:
            axs[1, 1].semilogy(wake_hist['iteration'], wake_hist['error'], '-o', ms=3)
            axs[1, 1].set_title('Free-wake convergence')
            axs[1, 1].set_xlabel('Wake iteration')
            axs[1, 1].set_ylabel('Max node displacement (m)')
            axs[1, 1].grid(True)
        else:
            axs[1, 1].axis('off')
    except Exception:
        axs[1, 1].axis('off')

    # Lift / drag distribution
    try:
        plot_blade_overlay(axs[1, 2], r_control, Lift_dist, 'Lift', style='-o')
        plot_blade_overlay(axs[1, 2], r_control, Drag_dist, 'Drag', style='-s')
        finish_axis(axs[1, 2], 'Section lift and drag', 'Force per unit span (N/m)')
    except Exception:
        pass

    plt.tight_layout()
    plt.show()

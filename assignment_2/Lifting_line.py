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
        self.altitude = 2000  # meters
        self.incidence = 0
        self.rotor_yaw = 0
        
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
            exponent_tip = -(self.n_blades / 2) * ((1 - mu) / mu) * np.sqrt(1 + (mu**2 * lambda_local**2) / ((1 - a)**2))
            #exponent_tip = np.clip(exponent_tip, -10, 0)
            f_tip = (2 / np.pi) * np.arccos(np.exp(exponent_tip))
        except:
            f_tip = 1.
        
        try:
            exponent_root = -(self.n_blades / 2) * ((mu - mu_root) / mu) * np.sqrt(1 + (mu**2 * lambda_local**2) / ((1 - a)**2))
            #exponent_root = np.clip(exponent_root, -10, 0)
            f_root = (2 / np.pi) * np.arccos(np.exp(exponent_root))
        except:
            f_root = 1.0
        
        F_total = f_tip * f_root
        #F_total = max(F_total, 0.0001)
        
        return F_total
    
    def biot_savart(self,X1,X2,Xp,gamma): #TODO: Add CORE like in answer_from_the_github_page.js
        eps=1e-6
        R1=np.sqrt((Xp[0]-X1[0])**2+(Xp[1]-X1[1])**2+(Xp[2]-X1[2])**2)
        R1=max(R1,eps)
        R2=np.sqrt((Xp[0]-X2[0])**2+(Xp[1]-X2[1])**2+(Xp[2]-X2[2])**2)
        R2=max(R2,eps)
        R12x=(Xp[1]-X1[1])*(Xp[2]-X2[2])-(Xp[2]-X1[2])*(Xp[1]-X2[1])
        R12y=-(Xp[0]-X1[0])*(Xp[2]-X2[2])+(Xp[2]-X1[2])*(Xp[0]-X2[0])
        R12z=(Xp[0]-X1[0])*(Xp[1]-X2[1])-(Xp[1]-X1[1])*(Xp[0]-X2[0])
        R12sqrt=R12x**2+R12y**2+R12z**2
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



    def Lifting_line(self, resolution=100,a_ind_wake=0,tend=5, tolerance=1e-6, max_iterations=1000, spacing='linear', use_prandtl=True, track_convergence=False, plot_geometry=False):
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
        r_control_abs = self.r_stations_abs#[:-1] + self.dr/2
        r_control_norm = r_control_abs/self.radius

        twist_stations = []
        chord_norm_stations = []
        for r_norm in r_control_norm:
            if r_norm >= self.blade_start_fraction:
                twist_stations.append(-50 * r_norm + 35 + self.collective_blade_pitch + self.collective_blade_pitch_location * 50 - 35 )
                chord_norm_stations.append(0.18 - 0.06 * r_norm)
            else:
                twist_stations.append(0)
                chord_norm_stations.append(0)

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
                for i in range(len(r_control_abs)-1):
                    r_in=self.r_stations_abs[i]
                    r_out=self.r_stations_abs[i+1]
                    r_mid=r_control_abs[i]
                    cp=rot_yz([0,r_mid, 0],angle_rotation)
                    controlpoints.append(cp)
                    panels.append(i)

                    filaments=[]
                    filaments.append((rot_yz([0,r_in, 0],angle_rotation),rot_yz([0,r_out, 0],angle_rotation)))

                    chord_in = chord_norm_stations[i] * self.radius
                    twist_in = twist_stations[i]
                    twist_in_rad = np.radians(twist_in)
                    x_te_in = chord_in * np.sin(-twist_in_rad)
                    z_te_in = -chord_in * np.cos(twist_in_rad)
                    filaments.append((rot_yz([x_te_in, r_in, z_te_in], angle_rotation), rot_yz([0, r_in, 0], angle_rotation)))

                    for j in range(len(theta_array) - 1):
                        xt = filaments[-1][0][0]; yt = filaments[-1][0][1]; zt = filaments[-1][0][2]
                        dy = (np.cos(-theta_array[j+1]) - np.cos(-theta_array[j])) * r_in
                        dz = (np.sin(-theta_array[j+1]) - np.sin(-theta_array[j])) * r_in
                        dx = (theta_array[j+1] - theta_array[j]) / (np.pi / (self.radius * self.J))*(1+a_ind_wake)
                        dx, dy, dz = rot_yz([dx, dy, dz], angle_rotation)
                        filaments.append((np.array([xt+dx, yt+dy, zt+dz]), np.array([xt, yt, zt])))


                    chord_out = chord_norm_stations[i+1] * self.radius
                    twist_out = twist_stations[i+1]
                    twist_out_rad = np.radians(twist_out)
                    x_te_out = chord_out * np.sin(-twist_out_rad)
                    z_te_out = -chord_out * np.cos(twist_out_rad)
                    filaments.append((rot_yz([0, r_out, 0], angle_rotation), rot_yz([x_te_out, r_out, z_te_out], angle_rotation)))

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
                cps = np.array(controlpoints)
                if cps.size:
                    ax3.scatter(cps[:, 0], cps[:, 1], cps[:, 2], c='r', marker='o', label='control points')

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

            u_ind=MatrixU @ Gamma
            v_ind=MatrixV @ Gamma
            w_ind=MatrixW @ Gamma

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

                a_temp[icp] = (-u_ind[icp] + vrot[0]) / (self.U_inf + 1e-12)
                alpha_temp[icp] = np.rad2deg(theta_rad - phi)
                phi_temp[icp] = np.rad2deg(phi)
                aline_temp[icp] = (vazim/(radialposition*self.omega) - 1)
                Fnorm_temp[icp] = cl * 0.5 * self.rho * V_effective**2 * chord_abs * np.cos(phi) + 0.5 * self.rho * V_effective**2 * chord_abs * cd_interp(alpha) * np.sin(phi)
                Ftan_temp[icp] = cl * 0.5 * self.rho * V_effective**2 * chord_abs * np.sin(phi) - 0.5 * self.rho * V_effective**2 * chord_abs * cd_interp(alpha) * np.cos(phi)


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

        return a_temp, aline_temp, Fnorm_temp, Ftan_temp, GammaNew, converged_iter, self.convergence_history, r_control_abs[1:] + self.dr/2, alpha_temp, phi_temp

        

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
            
            if r_norm <= self.blade_start_fraction:
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
                V_axial = self.U_inf * (1 - a)
                V_tangential = omega * r_abs * (1 + a_prime)
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
                
                # Corrected forces
                F_azimuth = (lift * np.sin(phi) - self.drag * np.cos(phi)) * F_prandtl
                F_axial = (lift * np.cos(phi) + self.drag * np.sin(phi)) * F_prandtl
                
                # Thrust coefficient

                A_a = 2 * np.pi * r_abs * self.dr[i-1]
                C_T = (F_axial * self.n_blades * self.dr[i-1]) / (0.5 * self.rho * self.U_inf**2 * A_a)
                if i==20 and iter_count%2==0:
                    self.CT_conv_list.append(C_T)
                    self.CT_conv_ind.append(iter_count)

                
                # Apply Glauert correction for axial induction
                a_calc = self._apply_glauert_correction(C_T) # Lucas: Is this necessary? Our propellor is not heavily loaded right?

                a_calc = np.clip(a_calc, 0, 0.95)
                # Azimuthal induction
                a_prime_calc = (F_azimuth * self.n_blades) / (2 * self.rho * (2 * np.pi * r_abs) * self.U_inf**2 * (1 - a_calc) * omega * r_abs)
                
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
        
        # Total coefficients
        self.CT = dT_total / (0.5 * self.rho * self.U_inf**2 * A_disk)
        self.CQ = dQ_total / (0.5 * self.rho * self.U_inf**2 * A_disk * self.radius)
        self.CP = self.CQ * omega * self.radius / self.U_inf



if __name__ == "__main__":
    
    bem = BEM(J=1.6, radius=0.7, n_blades=6, U_inf=60)

    tend=0.1
    dt=0.005
    bem.tlst=np.arange(0,tend,dt)
    # Uwake=10
    # print(bem.calc_ind_filiment([0,0,0.8],0.4))
    output = bem.Lifting_line(resolution=40, track_convergence=True, spacing='cosine')

    # Unpack outputs
    a_out, aline_out, Fnorm_out, Ftan_out, Gamma_out, conv_iter, conv_hist, r_control, alpha_out, phi_out = output

    blade_count = bem.n_blades
    station_count = len(r_control) + 1
    compare_with_bem = True

    r_l = r_control[1:]
    A_disk = np.pi * bem.radius**2

    C_T_LL = 0
    C_Q_LL = 0
    for i in range(len(r_l)):
        C_T_LL += (Fnorm_out[i+1] * blade_count * bem.dr[i]) / (0.5 * bem.rho * bem.U_inf**2 * A_disk)
        C_Q_LL += (Ftan_out[i+1] * blade_count * r_l[i] * bem.dr[i]) / (0.5 * bem.rho * bem.U_inf**2 * A_disk * bem.radius)

    C_P_LL = C_Q_LL * bem.omega * bem.radius / bem.U_inf
    print(f"Total thrust coefficient from lifting line: {C_T_LL:.4f}")
    print(f"Total torque coefficient from lifting line: {C_Q_LL:.4f}")
    print(f"Total power coefficient from lifting line: {C_P_LL:.4f}")


    def plot_blade_overlay(ax, x_values, y_values, label_prefix='', style='-o'):
        x_values = np.asarray(x_values)
        y_values = np.asarray(y_values)

        # remove r_index == 0 control points
        x_masked = x_values[1:]

        if station_count > 0 and len(y_values) == blade_count * (station_count - 1):
            blade_series = [np.asarray(y_values[i * (station_count - 1):(i + 1) * (station_count - 1)])[1:] for i in range(blade_count)]
            # if all blades identical, plot a single line
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
        # add vertical line showing blade start location
        try:
            blade_root_r = bem.blade_start_fraction * bem.radius
            ax.axvline(blade_root_r, color='k', linestyle='--', linewidth=1, label='blade start')
        except Exception:
            pass
        ax.legend()
        ax.grid(True)

    # Plot results
    bem_r_abs = None
    if compare_with_bem:
        # run BEM blade-element solver for comparison
        bem.blade_element(resolution=100, use_prandtl=False)
        bem_r_abs = np.array(bem.r_R_list) * bem.radius


    fig, axs = plt.subplots(2, 3, figsize=(15, 8))

    # Circulation
    try:
        plot_blade_overlay(axs[0, 0], r_control, Gamma_out * bem.n_blades * np.pi * bem.omega / bem.U_inf**2, 'Gamma')
        finish_axis(axs[0, 0], 'Nondimensional Circulation vs radius', r'$\frac{\Gamma N_\text{blades} \pi \Omega}{U_\infty^2}$')
        # overlay BEM circulation
        if compare_with_bem:
            try:
                if bem_r_abs is not None and len(bem.circulation_list) > 0:
                    axs[0, 0].plot(bem_r_abs, np.array(bem.circulation_list) * bem.n_blades * np.pi * bem.omega / bem.U_inf**2, '--k', label='BEM')
                    axs[0, 0].legend()
            except Exception:
                pass
    except Exception:
        pass

    # Axial and azimuthal induction
    try:
        plot_blade_overlay(axs[0, 1], r_control, a_out, 'a')
        plot_blade_overlay(axs[0, 1], r_control, aline_out, "a'", style='-s')
        finish_axis(axs[0, 1], 'Induction factors', 'Induction factor')
        # overlay BEM induction
        if compare_with_bem:
            try:
                if bem_r_abs is not None and len(bem.a_list) > 0:
                    axs[0, 1].plot(bem_r_abs, bem.a_list, '--k', label='BEM a')
                if bem_r_abs is not None and len(bem.a_prime_list) > 0:
                    axs[0, 1].plot(bem_r_abs, bem.a_prime_list, ':k', label="BEM a'")
                axs[0, 1].legend()
            except Exception:
                pass
    except Exception:
        pass

    # Forces
    try:
        plot_blade_overlay(axs[1, 0], r_control, Fnorm_out/(0.5 * bem.rho * bem.U_inf**2 * bem.radius), 'Fnorm')
        plot_blade_overlay(axs[1, 0], r_control, Ftan_out/(0.5 * bem.rho * bem.U_inf**2 * bem.radius), 'Ftan', style='-s')
        finish_axis(axs[1, 0], 'Section forces', r'Force per unit span coefficient $ C_F = \frac{1}{\frac{1}{2} \rho U_\infty^2 R}\frac{dF}{dr}$')
        # overlay BEM forces
        if compare_with_bem:
            try:
                if bem_r_abs is not None and len(bem.F_axial_list) > 0:
                    axs[1, 0].plot(bem_r_abs, np.array(bem.F_axial_list)/(0.5 * bem.rho * bem.U_inf**2 * bem.radius), '--k', label='BEM F_axial')
                if bem_r_abs is not None and len(bem.F_azimuth_list) > 0:
                    axs[1, 0].plot(bem_r_abs, np.array(bem.F_azimuth_list)/(0.5 * bem.rho * bem.U_inf**2 * bem.radius), ':k', label='BEM F_azimuth')
                axs[1, 0].legend()
            except Exception as e:
                print(e)
                pass
    except Exception as e:
        print(e)
        pass

    # Angle of attack
    try:
        plot_blade_overlay(axs[0, 2], r_control, alpha_out, 'AoA', style='-^')
        plot_blade_overlay(axs[0, 2], r_control, phi_out, 'Flow angle', style='-s')
        # overlay BEM AoA
        if compare_with_bem:
            try:
                if bem_r_abs is not None and len(bem.alpha_list) > 0:
                    axs[0, 2].plot(bem_r_abs, bem.alpha_list, '--k', label='BEM AoA')
                    axs[0, 2].plot(bem_r_abs, bem.phi_list, ':k', label='BEM flow angle')
                    axs[0, 2].legend()
            except Exception:
                pass
        finish_axis(axs[0, 2], 'Angle of attack and Flow angle', f'AoA, $\phi$  (degrees)')
    except Exception:
        pass

    # Convergence history
    try:
        if conv_hist is not None and 'error' in conv_hist and len(conv_hist['error'])>0:
            axs[1, 1].semilogy(conv_hist['iteration'], conv_hist['error'])
            axs[1, 1].set_title('Convergence history')
            axs[1, 1].set_xlabel('Iteration')
            axs[1, 1].set_ylabel('Relative error')
            axs[1, 1].grid(True)
        else:
            axs[1, 1].axis('off')
    except Exception:
        axs[1, 1].axis('off')

    plt.tight_layout()
    plt.show()

    # bem.blade_element(resolution=100, use_prandtl=False)
    
    # plot(
    #     "Spanwise Distribution: Angle of Attack",
    #     bem.r_R_list, [bem.alpha_list],
    #     ["Angle of Attack (α)"],
    #     "r/R", "α (deg)"
    # )
    
    # plt.show()

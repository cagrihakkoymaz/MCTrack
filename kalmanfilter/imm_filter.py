import numpy as np
import sys
from utils.utils import norm_radian

class Imm:
    def __init__(self, models, model_trans, P_trans, U_prob):
        self.models = models
        self.P_trans = P_trans
        self.U_prob = U_prob
        self.model_trans = model_trans

        self.num_models = len(models)
        self.dim = models[0].model.F.shape[0]  # <-- fixed here
        self.x_mix=None
        self.p_mix=None

    @staticmethod
    def ctra_to_cv_state(x_ctra):
        """
        Convert CTRA state to CV state: [x, y, vx, vy]
        x_ctra = [x, y, yaw, v, omega, a]
        """
        x_ctra = x_ctra.flatten()  # ensure it's 1D
        #print("x_ctra",x_ctra)
        x = x_ctra[0]
        y = x_ctra[1]
        yaw = x_ctra[2]
        v = x_ctra[3]

        vx = v * np.cos(yaw)
        vy = v * np.sin(yaw)

        return np.array([[x], [y], [vx], [vy]])
    @staticmethod
    def cv_to_ctra_state(x_cv):
        """
        Convert CV state [x, y, vx, vy] to CTRA state [x, y, yaw, v, omega, a]
        Assumes yaw_rate (omega) and acceleration (a) = 0
        """
        x_cv = x_cv.flatten()
        #print("x_cv",x_cv)

        x = x_cv[0]
        y = x_cv[1]
        vx = x_cv[2]
        vy = x_cv[3]

        v = np.sqrt(vx**2 + vy**2)
        yaw = np.arctan2(vy, vx)
        #print("yaw old : ",yaw)
        yaw = norm_radian(np.arctan2(vy, vx))
        #print("yaw new : ",yaw)

        return np.array([[x], [y], [yaw], [v], [0.0], [0.0]])

    def filt(self, Z,track_id):
        #print("================filter start")     

        # setp1: input mix
        u = np.dot(self.P_trans.T, self.U_prob)
        mu = np.zeros((self.num_models, self.num_models))
        for i in range(self.num_models):
            for j in range(self.num_models):
                mu[i, j] = self.P_trans[i, j] * self.U_prob[i, 0] / u[j, 0];
        #print("mu",mu)
        X_mix = [np.zeros(model.X.shape) for model in self.models]
        self.x_mix=X_mix
        if(track_id==0):
            pass
            #print("measurement : ",Z)
            #print("mu: ", mu)

        for j in range(self.num_models):
            if(track_id==0):
                    pass
                    #print("self.models[i].X",self.models[j].X) 
            for i in range(self.num_models):
                if(track_id==0):
                    pass
                    #print(' mu[i, j]', mu[i, j])
                    #print("X_mix",X_mix) 
                #print('self.model_trans[j][i]',self.model_trans[j][i])

                #print(' self.models[i].X)', self.models[i].X)
                #print(' mu[i, j]', mu[i, j])
                #print("i ;",i," j : ",j)
                #print("=========product=======",np.dot(self.model_trans[j][i],
                #                   self.models[i].X))
                #if self.models[i].X.ndim == 1:
                #    self.models[i].X = self.models[i].X.reshape(-1, 1)
 
                original=np.dot(self.model_trans[j][i],
                self.models[i].X)
                #print("original",original) 
                #print("shape",original.shape)   # Output: (4, 1)
                #print("ndim",original.ndim)    #

                debug=False

                if(debug):
                    X_mix[j] += np.dot(self.model_trans[j][i],
                                    self.models[i].X) * mu[i, j] 
                else:

                    if(i==0 and j==1):

                        # print("converted reshaped: ",self.cv_to_ctra_state(original).reshape(-1, 1))
                        # print("shape",self.cv_to_ctra_state(original).reshape(-1, 1).shape)   # Output: (4, 1)
                        # print("ndim",self.cv_to_ctra_state(original).reshape(-1, 1).ndim)   # Output: (4, 1)
                        # print("converted original: ",self.cv_to_ctra_state(original))           
                        # print("shape",self.cv_to_ctra_state(original).shape)    #
                        # print("ndim",self.cv_to_ctra_state(original).ndim)    #
                        if self.models[i].X.ndim == 1:
                            #print("---------case 1")

                            X_mix[j] += self.cv_to_ctra_state(original).flatten()  * mu[i, j]
                        else:
                            #print("-----------case 2")

                            X_mix[j] += self.cv_to_ctra_state(original).reshape(-1, 1) * mu[i, j]

                            #print("converted",self.cv_to_ctra_state(original).reshape(-1, 1))
                            
                    elif(i==1 and j==0):
                        # print("converted reshaped",self.ctra_to_cv_state(original).reshape(-1, 1))     
                        # print("shape",self.ctra_to_cv_state(original).reshape(-1, 1).shape)   # Output: (4, 1)
                        # print("ndim",self.ctra_to_cv_state(original).reshape(-1, 1).ndim)   # Output: (4, 1)
                        # print("converted original",self.ctra_to_cv_state(original))
                        # print("shape",self.ctra_to_cv_state(original).shape)    #
                        # print("ndim",self.ctra_to_cv_state(original).ndim)    #
                        if self.models[i].X.ndim == 1:
                            #print("---------------case 3")

                            X_mix[j] += self.ctra_to_cv_state(original).flatten() * mu[i, j]

                        else:
                            #print("-----------------case 4")

                            X_mix[j] += self.ctra_to_cv_state(original).reshape(-1, 1) * mu[i, j]

                        
                        #print("converted",self.ctra_to_cv_state(original).reshape(-1, 1))
                    else: 
                        
                        X_mix[j] += np.dot(self.model_trans[j][i],
                                        self.models[i].X) * mu[i, j] 
                        
                    if(track_id==0):
                        #print("X_mix after one",X_mix) 
                        pass

        #print("X_mix",X_mix)

        P_mix = [np.zeros(model.P.shape) for model in self.models]
        if(track_id==0):
            print(" self.models[i].P", np.diag(self.models[0].P))
        for j in range(self.num_models):
            for i in range(self.num_models):


                P = self.models[i].P + np.dot((self.models[i].X - X_mix[i]),
                                              (self.models[i].X - X_mix[i]).T)
                P_mix[j] += mu[i, j] * np.dot(np.dot(self.model_trans[j][i], P),
                                              self.model_trans[j][i].T)

        if(track_id==0):

            print(" P_mix updated", np.diag(P_mix[0]))

        '''
        for j in range(self.num_models):
            for i in range(self.num_models):
                # Transform both mean and covariance to model j space
                if i == 0 and j == 1:
                    x_ij = self.cv_to_ctra_state(np.dot(self.model_trans[j][i], self.models[i].X))
                    P_ij = self.model_trans[j][i] @ self.models[i].P @ self.model_trans[j][i].T
                elif i == 1 and j == 0:
                    x_ij = self.ctra_to_cv_state(np.dot(self.model_trans[j][i], self.models[i].X))
                    P_ij = self.model_trans[j][i] @ self.models[i].P @ self.model_trans[j][i].T
                else:
                    x_ij = np.dot(self.model_trans[j][i], self.models[i].X)
                    P_ij = self.model_trans[j][i] @ self.models[i].P @ self.model_trans[j][i].T

                delta = x_ij - X_mix[j]
                P_mix[j] += mu[i, j] * (P_ij + delta @ delta.T)
        '''
        self.p_mix=P_mix
        
        ## step2: filt
        for j in range(self.num_models):
            #X_mix[j]= np.nan_to_num(X_mix[j], nan=0.0)
            if(track_id==0):
                #print("before kalman")
                #print("self.models[j].X ",self.models[j].X )
                #print("P model",self.models[j].P )
                pass
            self.models[j].X = X_mix[j]
            if(track_id==0):
                #print("is there problem in assignment")
                #print("self.models[j].X ",self.models[j].X )
                #print("P model mixed",self.models[j].P )

                pass
            self.models[j].P = P_mix[j]
            #self.models[j].assign_mixed_values()
            self.models[j].filt(Z,track_id)
            if(track_id==0):
                #print("after kalman")
                #print("self.models[j].X ",self.models[j].X )
                #print("P model updated",self.models[j].P )
                pass

        ### step3: update probability
        for j in range(self.num_models):
            Z = np.array(Z).reshape(-1, 1)

            mode = self.models[j]

            D = Z - np.dot(mode.H, mode.X_pre)
            #print("mode.H",mode.H)        
            if(track_id==0):
                pass
                #print("mode.X_pre",mode.X_pre)
            mode.X_pre = np.nan_to_num(mode.X_pre, nan=0.0)

            if np.isnan(mode.X_pre[0, 0]):
                print("Value is NaN. Exiting.")
                sys.exit(1)
            #print("Z",Z)
            #print("model guess",np.dot(mode.H, mode.X_pre))
            #print("D",D)
            S = np.dot(np.dot(mode.H, mode.P_pre), mode.H.T) + mode.R
            n = Z.shape[0]  # Number of measurements (e.g., 2 for x, y)

            # Regularize S
            S += 1e-6 * np.eye(S.shape[0])

            try:
                S_inv = np.linalg.inv(S)
                det_S = np.linalg.det(S)

                if det_S <= 0 or np.isnan(det_S):
                    print(f"Invalid determinant encountered: det(S)={det_S}, replacing with small positive.")
                    det_S = 1e-6  # Avoid zero or negative determinant

                # Multivariate Gaussian likelihood
                norm_const = 1.0 / (np.sqrt((2 * np.pi)**n * det_S))
                exponent = -0.5 * (D.T @ S_inv @ D)
                Lambda = norm_const * np.exp(exponent)

            except np.linalg.LinAlgError:
                print("S matrix is singular. Skipping this model.")
                Lambda = 0.0


            #print("Lambda : ",Lambda)
            #print("u[j, 0] : ",u[j, 0])
            if(track_id==0):
                pass
                #print("D : ",D)
                #print("S : ",S)
                #print("Lambda",Lambda)


            self.U_prob[j, 0] = Lambda * u[j, 0]
            #print("self.U_prob[j, 0] : ",self.U_prob[j, 0])
            if np.isnan(self.U_prob[j, 0]):
                self.U_prob[j, 0]=0.5
            if np.isnan(self.U_prob[j, 0]):
                print("Value is NaN. Exiting.")
                sys.exit(1)
        #print("self.U_prob",self.U_prob) 
        #print("np.sum(self.U_prob)",np.sum(self.U_prob))    

        self.U_prob = self.U_prob / np.sum(self.U_prob)

        if(track_id==0):
            print("after update self.U_prob",self.U_prob) 
            if (self.U_prob[0]>0.5):
                print("-------------------------------CV mode on-------------------------------------")
            else   :
                print("-------------------------------CTRA mode on-------------------------------------")
    

        return self.U_prob

    def get_fused_state(self):
        # Choose CV (4D) as fusion space
        fused_state = np.zeros((self.dim, 1))
        for i in range(self.num_models):
            X_i = self.models[i].X
            #print(i, " model state guess is : ",X_i)
            if X_i.ndim == 1:
                X_i = X_i.reshape(-1, 1)
            if i == 1:  # assuming model 1 is CTRA
                projected_state = self.ctra_to_cv_state(X_i)
            else:
                projected_state = self.model_trans[0][i] @ X_i


            #print("projected_state",projected_state)
            #print("self.U_prob[i, 0]",self.U_prob[i, 0])

            if np.isnan(self.U_prob[i, 0]):
                self.U_prob[i, 0]=0.5
            fused_state += self.U_prob[i, 0] * projected_state
            #print("fused_state",fused_state)
    
        #print("fused_state.flatten()",fused_state.flatten())  
        if np.isnan(fused_state.flatten()[0]):
                sys.exit(1)  
        return fused_state.flatten()



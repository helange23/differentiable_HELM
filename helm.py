import tensorflow as tf
import numpy as np


class dHELM:
    
    def __init__(self, Y, slack_bus = 146):
        
        self.Y = Y
        self.Y_ns = np.concatenate([Y[:slack_bus], Y[slack_bus+1:]], 0)
        self.y_slack = self.Y_ns[:,slack_bus] #slack row
        self.Y_ns = np.concatenate([self.Y_ns[:,:slack_bus],
                                    self.Y_ns[:,slack_bus+1:]], 1) #Y without slack
        
        self.Y_inv = np.linalg.inv(self.Y_ns) #inverse of reduced Y
        self.slack_bus = slack_bus
        
        comp = lambda x: tf.complex(np.real(x), np.imag(x))
        self.Y_inv, self.Y_ns = comp(self.Y_inv), comp(self.Y_ns)
        self.y_slack = comp(self.y_slack)
        self.Y = comp(self.Y)
        
        
    def remove_slack(self, S):
        S1 = S[:,:self.slack_bus]
        S2 = S[:,self.slack_bus+1:]
        
        if type(S) == np.ndarray:
            return np.concatenate([S1,S2],axis=-1)
        
        return tf.concat([S1,S2],axis=-1)
    
    
    def get_voltage(self, S, slack_v, iterations = 50):

        Y_inv, Y_bus, y_slack = self.Y_inv, self.Y, self.y_slack
        slack_bus = self.slack_bus
        
        batch_num = S.shape[0]
        exp = lambda x: tf.expand_dims(x,1)
        
        Y_slack = tf.repeat([y_slack],batch_num,axis=0)
        
        slack_v = tf.complex(tf.cast(tf.expand_dims(slack_v,-1),dtype=tf.float64),
                             tf.zeros((batch_num,1),dtype=tf.float64))
            

        c = exp(tf.einsum('ji,aj->ai',Y_inv, -Y_slack*slack_v))
        d = 1./c


        for n in range(iterations-1):
            RHS = tf.math.conj(S)*tf.math.conj(d[:,-1,:])
            new_c = exp(tf.einsum('ij,aj->ai', Y_inv, RHS))
            c = tf.concat([c,new_c], axis=1)
            new_d = -tf.reduce_sum(tf.reverse(c[:,1:,:],[1])*d,axis=1, keepdims=True)/exp(c[:,0,:])
            d = tf.concat([d,new_d], axis=1)

            
        def pade_approx_1(an):
            
            m = int(iterations/2)
            N = iterations-1
            n = int(iterations-1-iterations/2)
        
            an = exp(an)
            
            A = tf.eye(N+1, n+1, [batch_num], dtype=tf.complex128)
            B = [tf.zeros((batch_num,1,m),dtype=tf.complex128)]
            
            for row in range(1, m+1):
                z = tf.complex(tf.zeros((batch_num,1,m-row),dtype=tf.float64),
                               tf.zeros((batch_num,1,m-row),dtype=tf.float64))
                B.append(tf.concat([-tf.reverse(an[:,:,:row],[2]),z],axis=2))
            for row in range(m+1, N+1):
                B.append(-tf.reverse(an[:,:,row-m:row],[2]))
            
            B = tf.concat(B, axis=1)
            C = tf.concat([A,B], axis=2)
            pq = tf.linalg.solve(C, tf.transpose(an,[0,2,1]))[:,:,0]
            
            p = tf.reduce_sum(pq[:,:n+1],axis=-1)
            q = 1.0 + tf.reduce_sum(pq[:,n+1:],axis=-1)
            
            return p/q
        
        v = tf.transpose(tf.map_fn(pade_approx_1, tf.transpose(c,[2,0,1])))
        
        v = tf.concat([v[:,:slack_bus],slack_v,v[:,slack_bus:]],axis=1)
        v_mat = tf.einsum('ai,aj->aij', tf.math.conj(v), v)
        S_out = tf.math.conj(tf.reduce_sum(v_mat*Y_bus,axis=-1))
        
        S_ns = self.remove_slack(S_out)
        err = tf.reduce_max(tf.abs(S_ns - S),axis=-1)
        
        return v, S_out, tf.math.log(err), c
import tensorflow as tf
import numpy as np


def remove_slack(S,slack_bus):
    S1 = S[:,:slack_bus]
    S2 = S[:,slack_bus+1:]
    return tf.concat([S1,S2],axis=-1)


def get_voltage(S, slack_v, Y_inv, Y_bus, Y_slack, slack_bus, iterations = 50):
    #S = S/g.baseMVA
    
    batch_num = S.shape[0] #test
    
    comp = lambda x: tf.complex(np.real(x), np.imag(x))
    exp = lambda x: tf.expand_dims(x,1)
    
    #Y_inv = comp(Yinv)
    #Y_bus = comp(Ybus)
    Y_slack = comp(np.repeat([Y_slack],batch_num,axis=0))
    
    slack_v = tf.complex(tf.cast(tf.expand_dims(slack_v,-1),dtype=tf.float64),
                         tf.zeros((batch_num,1),dtype=tf.float64))
        
    c = exp(tf.einsum('ji,aj->ai',Y_inv, -Y_slack*slack_v))
    d = 1./c
    
    for n in range(iterations-1):
        RHS = tf.conj(S)*tf.conj(d[:,-1,:])
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
        pq = tf.matrix_solve(C, tf.transpose(an,[0,2,1]))[:,:,0]
        
        p = tf.reduce_sum(pq[:,:n+1],axis=-1)
        q = 1.0 + tf.reduce_sum(pq[:,n+1:],axis=-1)
        
        return p/q
    
    v = tf.transpose(tf.map_fn(pade_approx_1, tf.transpose(c,[2,0,1])))
    
    v = tf.concat([v[:,:slack_bus],slack_v,v[:,slack_bus:]],axis=1)
    v_mat = tf.einsum('ai,aj->aij', tf.conj(v), v)
    S_out = tf.conj(tf.reduce_sum(v_mat*Y_bus,axis=-1))
    
    S_ns = remove_slack(S_out, slack_bus)
    err = tf.reduce_max(tf.abs(S_ns - S),axis=-1)
    
    return v, S_out, tf.log(err), c
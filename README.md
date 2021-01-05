# Differentiating through Holomorphic Embeddings (HELM)
This repository contains the code to compute voltages and their derivatives based on the Holomorphic Embedded Load Flow Method introduced by Trias et al.
This code was used in the paper ["Learning to Solve AC Optimal Power Flow by Differentiating through Holomorphic Embeddings"](https://arxiv.org/pdf/2012.09622.pdf).

A video abstract of the paper will follow soon.

Usage example (example files extracted from the IEEE 200bus Illinois case are contained in the repository):

```python
    from helm import dHELM

    Y = np.load('Y.npy')
    Sg = np.load('Sg.npy')
    Sg = tf.complex(Sg[:,0], Sg[:,1])
    Sd = np.load('Sd.npy')
    Sd = tf.complex(Sd[:,0], Sd[:,1])

    h = dHELM(Y)

    with tf.GradientTape(persistent=True) as g:

        g.watch(Sg)

        S = Sd-Sg
        S_ns = h.remove_slack(S[None])

        v, S_out, err, c = h.get_voltage(S_ns, 1.0)

        dSg_dS = g.gradient(S_out[:,146], Sg)
        dSg_dv = g.gradient(v[:,23], Sg)

        print(dSg_dS)
        print(dSg_dv)
``` 

Note that S has a leading batch dimension, i.e. it is a 2D matrix.

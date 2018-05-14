import tensorflow as tf
import tensorlayer.layer as tl
import numpy as np
def prune_network(network, sess, candidate)
    for index,element in enumerate(candidate):
        var_tensor = tl.get_variables_with_name("res"+str(index)+"/"+"c1")
        w = var_tensor[0]
        b = var_tensor[1]
        w_next = tl.get_variables_with_name("res"+str(index)+"/"+"c2")[0]
                
        w_np = w.eval()
        b_np = b.eval()
        w_next_np = w_next.eval()
                
        w_np = np.delete(w_np, element, -1)
        b_np = np.delete(b_np, element, 0)
        w_next_np = np.delete(w_next_np, element, -2)

        w_new = tf.convert_to_tensor(w_np)
        b_new = tf.convert_to_tensor(b_np)
        w_next_new = tf.convert_to_tensor(w_next_np)
                
        sess.run(tf.assign(w, w_new, False))
        sess.run(tf.assign(b, b_new, False))
        sess.run(tf.assign(w_next, w_next_new, False))


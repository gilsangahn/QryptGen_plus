# lorenz key 3000
# WGAN + anticorr + entropy

import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit
import pennylane as qml
from pennylane.templates import StronglyEntanglingLayers
import optax
import time
from PIL import Image
from datetime import datetime, timedelta, timezone

print("JAX version:", jax.__version__)
print("Available devices:", jax.devices())

n_qubits = 8
n_ancillas = 1
layers = 10
n_generators = 7
n_epochs = 50
batch_size = 25
g_lr = 0.01
d_lr = 0.0002
n_critic = 10
lambda_gp = 10
lambda_1 = 0.6
lambda_2 = 0.3
patch_shape = (4, 28)

data_qubits = n_qubits - n_ancillas
positions = [(0, 0), (4, 0), (8, 0), (12, 0), (16, 0), (20, 0), (24, 0)]

dev = qml.device("default.qubit", wires=n_qubits, shots=None)

@qml.qnode(dev, interface="jax", diff_method="backprop")
def patch_circuit(latent_vector, weights):
    for i in range(n_qubits):
        qml.RY(latent_vector[i], wires=i)
    StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return qml.probs(wires=range(n_qubits))

batched_patch_circuit = jax.vmap(patch_circuit, in_axes=(0, None))

def generator_forward(latents, all_params):
    image = jnp.zeros((latents.shape[0], 1, 28, 28))
    for idx, (py, px) in enumerate(positions):
        full_probs = batched_patch_circuit(latents, all_params[idx])  
        probs_given_ancilla_0 = full_probs[:, :2**data_qubits]        
        post_probs = probs_given_ancilla_0 / jnp.sum(probs_given_ancilla_0, axis=1, keepdims=True)
        patch_probs = post_probs[:, :112]
        patch = ((patch_probs / jnp.max(patch_probs, axis=1, keepdims=True)) - 0.5) * 2
        patch = patch.reshape((latents.shape[0], 1, *patch_shape))
        image = image.at[:, :, py:py + patch_shape[0], px:px + patch_shape[1]].set(patch)
    return image

def correlation_loss(images):
    img = images.squeeze(1)
    vert_diff = img[:, 1:, :] - img[:, :-1, :]
    horiz_diff = img[:, :, 1:] - img[:, :, :-1]
    loss_v = jnp.mean(vert_diff**2)
    loss_h = jnp.mean(horiz_diff**2)
    return loss_v + loss_h

def entropy_loss(images):
    probs = (images + 1) / 2
    probs = jnp.clip(probs, 1e-8, 1.0) 
    entropy = -probs * jnp.log2(probs) 
    avg_entropy = jnp.mean(entropy) * 8  
    return avg_entropy

def critic_forward(x, critic_params):
    x = x.reshape((x.shape[0], -1))
    x = jax.nn.leaky_relu(x @ critic_params["w1"] + critic_params["b1"], 0.2)
    x = jax.nn.leaky_relu(x @ critic_params["w2"] + critic_params["b2"], 0.2)
    return x @ critic_params["w3"] + critic_params["b3"]

def init_critic_params(key):
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        "w1": jax.random.normal(k1, (784, 512)) * 0.02,
        "b1": jnp.zeros(512),
        "w2": jax.random.normal(k2, (512, 256)) * 0.02,
        "b2": jnp.zeros(256),
        "w3": jax.random.normal(k3, (256, 1)) * 0.02,
        "b3": jnp.zeros(1)
    }

def compute_gp(critic_params, real, fake, rng_key):
    epsilon = jax.random.uniform(rng_key, shape=(real.shape[0], 1, 1, 1))
    interpolated = epsilon * real + (1 - epsilon) * fake
    def single_interp_gp(x):
        def output(xi):
            return critic_forward(xi[None, ...], critic_params).squeeze()
        grads = jax.grad(output)(x)
        norm = jnp.linalg.norm(grads.reshape(-1))
        return (norm - 1.0)**2
    gp_values = jax.vmap(single_interp_gp)(interpolated)
    return jnp.mean(gp_values)

def denorm(x):
    return ((x + 1) / 2).clip(0, 1)

fixed_key = jax.random.PRNGKey(9999)
fixed_z = jax.random.uniform(fixed_key, (batch_size, n_qubits))


KST = timezone(timedelta(hours=9)) 
now_kst = datetime.now(KST)
now_str = now_kst.strftime("%Y-%m-%d %H:%M:%S")
with open("QryptGen_p_log.txt", "w") as f:
    now_kst = datetime.now(timezone(timedelta(hours=9)))
    f.write(f"{now_kst.strftime('%Y-%m-%d %H:%M:%S')} === QryptGen+ Log Start ===\n")
    f.write(f"lambda_1 (AntiCorr weight): {lambda_1}, lambda_2 (Entropy weight): {lambda_2}\n\n")    

def train():
    master_key = jax.random.PRNGKey(time.time_ns() % (2**32 - 1))
    master_key, gen_key, critic_key = jax.random.split(master_key, 3)
    gen_param_keys = jax.random.split(gen_key, n_generators)

    out_key = "QG_key_images"
    out_results = "QG_results10"
    os.makedirs(out_key, exist_ok=True)
    os.makedirs(out_results, exist_ok=True)

    data_np = np.load(os.path.join(out_key, "lorenz_key.npy"))
    data = jnp.array(data_np / 255.0).reshape((-1, 1, 28, 28))
    n_batches = data.shape[0] // batch_size

    params = [0.01 * jax.random.normal(k, shape=(layers, n_qubits, 3)) for k in gen_param_keys]
    critic_params = init_critic_params(critic_key)

    g_opt = optax.adam(g_lr, b1=0.0, b2=0.9)
    d_opt = optax.adam(d_lr, b1=0.0, b2=0.9)
    g_state = g_opt.init(params)
    d_state = d_opt.init(critic_params)

    wasserstein_distance_history = []
    g_losses, d_losses, total_losses = [], [], []
    critic_fake, anti_correlation, entropy = [], [], []

    for epoch in range(n_epochs):
        epoch_start_time = time.time()
        for b in range(n_batches):
            z_key, master_key = jax.random.split(master_key)
            z = jax.random.uniform(z_key, (batch_size, n_qubits))
            batch = data[b*batch_size : (b+1)*batch_size]

            def d_loss_fn(d_params):
                fake = generator_forward(z, params)
                real_score = critic_forward(batch, d_params)
                fake_score = critic_forward(fake, d_params)
                gp = compute_gp(d_params, batch, fake, z_key)
                loss = jnp.mean(fake_score) - jnp.mean(real_score) + lambda_gp * gp
                return loss, fake

            (d_loss_val, fake), d_grads = jax.value_and_grad(d_loss_fn, has_aux=True)(critic_params)
            entropy_val_iter = entropy_loss(fake)
            anti_corr_val_iter = correlation_loss(fake)
            w_dist_iter = float(jnp.mean(critic_forward(batch, critic_params)) - jnp.mean(critic_forward(fake, critic_params)))

            entropy.append(float(entropy_val_iter))
            anti_correlation.append(float(anti_corr_val_iter))
            wasserstein_distance_history.append(w_dist_iter)

            updates, d_state = d_opt.update(d_grads, d_state)
            critic_params = optax.apply_updates(critic_params, updates)

            if b % n_critic == 0:
                        
                def g_loss_fn(g_params):
                     fake = generator_forward(z, g_params)
                     critic_fake = critic_forward(fake, critic_params).mean()
                     anti_corr = correlation_loss(fake)
                     entropy = entropy_loss(fake)
                     total_loss = -critic_fake - lambda_1 * anti_corr - lambda_2 * entropy
                     return total_loss, (critic_fake,)

                (g_loss_val, (critic_fake_val,)), g_grads = jax.value_and_grad(g_loss_fn, has_aux=True)(params)
                g_updates, g_state = g_opt.update(g_grads, g_state)
                params = optax.apply_updates(params, g_updates)

                g_losses.append(float(g_loss_val))
                d_losses.append(float(d_loss_val))
                critic_fake.append(float(critic_fake_val))
                total_losses.append(float(g_loss_val) + float(d_loss_val))

                log_line = (f"[Epoch {epoch+1}/{n_epochs}] [Batch {b+1}/{n_batches}], "
                            f"[G loss: {float(g_loss_val):.6f}], "f"[D loss: {float(d_loss_val):.6f}], "
                            f"[c_fake loss: {float(critic_fake_val):.6f}], "
                            f"[W: {float(w_dist_iter):.6f}], "
                            f"[Anti correlation: {float(anti_corr_val_iter):.6f}], "f"[Entropy: {float(entropy_val_iter):.6f}]\n")
                with open("QryptGen_p_log.txt", "a") as f:
                    f.write(log_line)

                fixed_img = generator_forward(fixed_z[:1], params)[0]
                img_np = np.array(denorm(fixed_img.squeeze())) * 255
                Image.fromarray(img_np.astype(np.uint8)).save(f"{out_results}/e{epoch+1}_b{b+5}.png")

            if b == n_batches - 1:
                elapsed_time = time.time() - epoch_start_time
                log_line = (f"[Epoch {epoch+1}/{n_epochs}] [Batch {b+1}/{n_batches}], "
                            f"[G loss: {float(g_loss_val):.6f}], "
                            f"[D loss: {float(d_loss_val):.6f}], "
                            f"[c_fake loss: {float(critic_fake_val):.6f}], "
                            f"[W: {float(w_dist_iter):.6f}], "
                            f"[Anti correlation: {float(anti_corr_val_iter):.6f}], "f"[Entropy: {float(entropy_val_iter):.6f}]"
                            f"[Time: {elapsed_time:.2f}s]\n\n")
                print(log_line.strip())
                with open("QryptGen_p_log.txt", "a") as f:
                    f.write(log_line)

    np.save(os.path.join(out_results, "wasserstein_distance.npy"), np.array(wasserstein_distance_history))
    np.save(os.path.join(out_results, "g_loss.npy"), np.array(g_losses))
    np.save(os.path.join(out_results, "d_loss.npy"), np.array(d_losses))
    np.save(os.path.join(out_results, "critic_fakenpy"), np.array(critic_fake))
    np.save(os.path.join(out_results, "anti_correlation.npy"), np.array(anti_correlation))
    np.save(os.path.join(out_results, "entropy.npy"), np.array(entropy)) 
    np.save(os.path.join(out_results, "total_loss.npy"), np.array(total_losses))
    np.save(os.path.join(out_results, "initial_generator_params.npy"), np.array(params))
    np.save(os.path.join(out_results, "initial_critic_params.npy"), critic_params)
    np.save(os.path.join(out_results, "fixed_latent_z.npy"), np.array(fixed_z))

train()

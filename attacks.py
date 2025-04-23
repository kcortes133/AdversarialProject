import torch
import numpy as np
# https://www.kaggle.com/code/alexine/adversarial-attacks/notebook#Attacks
def fgsm_attack(x_original, epsilon, gradient):
    # Get Gradient sign
    grad_sign = gradient.sign()
    # Add epsilon*grad_sign perturbation to the original input
    perturbation = epsilon*grad_sign
    x_perturbed = x_original + perturbation
    return x_perturbed, perturbation




epsilons = torch.Tensor([0.1, 0.05, 0.01, 0.001])
acc_results_non = dict()
verbose = False
N_VAL_SAMPLES = testset.data.shape[0]
model_no_gaussian.eval()
for eps in epsilons:
    correct_unperturbed = 0
correct_perturbed = 0
t0 = time.perf_counter()
for j, val_data in enumerate(validation_loader, 0):
### NOTE: IT WOULD BE MORE EFFICIENT TO ITERATE ONLY ONCE THROUGH THE DATA AND PERFORM ALL THE ATTACKS
    x, y_target = val_data
x, y_target = x.to(device), y_target.to(device)
x.requires_grad = True
output = model_no_gaussian(x)
y_pred = torch.argmax(output)

if y_pred == y_target:  # Only make attack on correctly classified samples
    correct_unperturbed += 1
# Calculate loss and gradient
loss = criterion(output, y_target)
grad = torch.autograd.grad(outputs=loss, inputs=x)[0]
model_no_gaussian.zero_grad()
perturbed_x, _ = fgsm_attack(x, epsilon=eps, gradient=grad)
perturbed_output = model_no_gaussian(perturbed_x)
y_pred_perturbed = torch.argmax(perturbed_output)
loss_perturbed = criterion(perturbed_output, y_target)
if y_pred_perturbed == y_target:
    correct_perturbed += 1
acc_before_attack = correct_unperturbed / N_VAL_SAMPLES
acc_after_attack = correct_perturbed / N_VAL_SAMPLES
print(f'\nFGSM Attack with epsilon = {eps:.5f} | Elapsed time: {time.perf_counter() - t0:.2f} seconds.')
print(
    f'Accuracy: Before the attack -> {100 * acc_before_attack:.2f}%\t|\tAfter the attack -> {100 * acc_after_attack:.2f}%')
acc_results_non[eps.item()] = acc_after_attack
acc_results_non[0] = acc_before_attack

# https://www.kaggle.com/code/bowaka/defcon31-granny-square-attack-in-details
def clip_perturbation(original_img, perturbed_img, epsilon):
    """
    Ensure that the perturbation is within the allowed epsilon
    """
    delta = perturbed_img - original_img
    # Clip the changes to [-epsilon, epsilon]
    clipped_delta = np.clip(delta, -epsilon, epsilon)
    # Apply the clipped perturbation to the original image
    clipped_adv_img = original_img + clipped_delta
    # Ensure that the values are valid (e.g., in [0,255] for images)
    clipped_adv_img = np.clip(clipped_adv_img, 0, 255)
    return clipped_adv_img.astype(int)

def gen_vertical_perturbation(img, epsilon, band_width):
    """This function create the vertical perturbation.
    img (arr): the original image to modify,in numpy array format
    epsilon (int): the maximum intensity of the perturbation to apply
    band_width (int): the size (in px) of the perturbation bands
    """
    adv_mask = img.copy()
    shape = img.shape
    idx_start = 0
    for band in range(3):
        while True:
            perturbation_value = np.random.randint(-epsilon, epsilon + 1)
            adv_mask[:, idx_start:idx_start + band_width, band] = adv_mask[:, idx_start:idx_start + band_width,
                                                                  band] + perturbation_value
            idx_start += band_width
            # When we reach the end of the image, we get out of the loop
            if idx_start + band_width >= shape[1]:
                break

    # Once the perturbated image is generated, we use our clipping function to restrain the modification
    adv_image = clip_perturbation(np.clip(img, 0, 255).astype(int), adv_mask, epsilon)

    return adv_image


epsilon = 20
band_width = 1
adv_image = gen_vertical_perturbation(img_arr, epsilon, band_width)

plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.title("Original image")
plt.imshow(img_arr)
plt.subplot(1, 2, 2)
plt.title("Perturbed image")
plt.imshow(adv_image)

def gen_squared_perturbation(original_image,
                             adv_image,
                             epsilon,
                             min_perturbation_size,
                             max_perturbation_size):
    """Add a squared perturbation to the current adversial image
    original_image (arr): the original image, used to bound our perturbation
    adv_image (arr): the current adversial image
    epsilon (int): the maximum perturbation according Linf distance
    min_perturbation_size (int): minimum size of the perturbation box
    max_perturbation_size (int): maximum size of the perturbation box
    """

    shape = original_image.shape

    adv_image_with_squared_perturbation = adv_image.copy()
    # I decided to randomize the perturbation width and the perturbation height
    pert_width = np.random.randint(min_perturbation_size, max_perturbation_size)
    pert_height = np.random.randint(min_perturbation_size, max_perturbation_size)

    # Draw a start index for the block we'll perturb. We need to make sure we are not out of bounds
    idx_x_start = np.random.randint(0, shape[1] - pert_width + 1)
    idx_y_start = np.random.randint(0, shape[0] - pert_height + 1)

    # Add perturbation
    perturbation = np.random.randint(-epsilon, epsilon + 1, 3)
    adv_image_with_squared_perturbation[idx_y_start:idx_y_start + pert_width,
    idx_x_start:idx_x_start + pert_height, :] = adv_image[idx_y_start:idx_y_start + pert_width,
                                                idx_x_start:idx_x_start + pert_height, :] + perturbation

    adv_image_with_squared_perturbation = clip_perturbation(original_image,
                                                            np.clip(adv_image_with_squared_perturbation.astype(int), 0,
                                                                    255),
                                                            epsilon)

    return adv_image_with_squared_perturbation


epsilon = 80
min_perturbation_size = 1
max_perturbation_size = 3
adv_image_with_squared_perturbation = gen_squared_perturbation(img_arr,
                                                               adv_image,
                                                               epsilon,
                                                               min_perturbation_size,
                                                               max_perturbation_size)

plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.title("Perturbed (vertical stripes) image")
plt.imshow(adv_image)
plt.subplot(1, 2, 2)
plt.title("Perturbed image (with a new squared perturbation)")
plt.imshow(adv_image_with_squared_perturbation)
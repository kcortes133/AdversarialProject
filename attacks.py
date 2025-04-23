import torch
def fgsm_attack(x_original, epsilon, gradient):
    # Get Gradient sign
    grad_sign = gradient.sign()
    # Add epsilon*grad_sign perturbation to the original input
    perturbation = epsilon*grad_sign
    x_perturbed = x_original + perturbation
    return x_perturbed, perturbation

)


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
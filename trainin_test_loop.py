import subprocess

# Number of training runs
num_training_runs = 10

for i in range(num_training_runs):
    print(f"Starting training run {i + 1}/{num_training_runs}...")

    # Call the training script
    training_result = subprocess.run(
        [r"C:\Users\aaron\PycharmProjects\jassager\.env3\Scripts\python.exe", "rl_train_model.py"],
        check=True
    )
    if training_result.returncode != 0:
        print(f"Training run {i + 1} failed. Stopping.")
        break

    print(f"Completed training run {i + 1}/{num_training_runs}. Starting testing...")

    # # Call the testing script
    # testing_result = subprocess.run(
    #     [r"C:\Users\aaron\PycharmProjects\jassager\.env3\Scripts\python.exe", "rl_model_test.py"],
    #     check=True
    # )
    # if testing_result.returncode != 0:
    #     print(f"Testing after training run {i + 1} failed. Stopping.")
    #     break
    #
    # print(f"Testing after training run {i + 1} completed successfully.")

print("All training and testing runs are complete.")

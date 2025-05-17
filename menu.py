import subprocess
import sys

def run_script(script_name):
    try:
        subprocess.run([sys.executable, script_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")

def main():
    while True:
        print("\n=== Chrome Dino RL Menu ===")
        print("1. Train Model")
        print("2. Evaluate Model")
        print("3. Test Environment")
        print("4. Exit")
        choice = input("Select an option: ").strip()

        if choice == "1":
            run_script("dqn_train.py")
        elif choice == "2":
            run_script("dqn_eval.py")
        elif choice == "3":
            run_script("test_script.py")
        elif choice == "4":
            print("Exiting.")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()

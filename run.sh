# create a virtual environment and run the main script
#!/bin/bash
# Check if the virtual environment directory exists
if [ ! -d "venv_master_thesis" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv_master_thesis
else
    echo "Virtual environment already exists."
fi

#activate virtual environment
source ./venv_master_thesis/bin/activate

# Install requirements
python3 -m pip install --upgrade pip
pip3 install -r requirements.txt

# Run the main script
python3 main.py
# Deactivate the virtual environment
deactivate

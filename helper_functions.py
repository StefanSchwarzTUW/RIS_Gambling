# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 08:19:49 2025

@author: Stefan Schwarz

Some helper functions

"""

import importlib.util
import subprocess
import sys

def install_if_missing(package_name):
    # Check if the package is installed
    if importlib.util.find_spec(package_name) is None:
        # Ask the user for permission
        response = input(f"The package '{package_name}' is not installed. Do you want to install it now? [y/n]: ").strip().lower()
        if response == "y":
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", package_name], check=True)
                print(f"✅ '{package_name}' installed successfully.")
            except subprocess.CalledProcessError:
                print(f"❌ Failed to install '{package_name}'.")
        else:
            print(f"⚠️ '{package_name}' is required but was not installed.")
    else:
        print(f"✅ '{package_name}' is already installed.")
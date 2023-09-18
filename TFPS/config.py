""" Stores code config and settings """

# Import necessary libraries and modules
import json

settings: {}
# Open and read the 'config.json' file, loading its content into the 'settings' dictionary
with open('config.json', 'r') as f:
    settings = json.load(f)

# Define a function named 'get_setting' that takes a 'key' as an argument
def get_setting(key):
    # Return the value associated with the provided 'key' from the 'settings' dictionary
    return settings[key]
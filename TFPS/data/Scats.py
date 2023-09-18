""" Interact with an SQLite database for SCATS traffic management system """

# Import necessary libraries and modules
import sqlite3
import numpy as np

# Import custom functions and classes
from config import get_setting


# Define a class for interacting with the SCATS database
class ScatsDB(object):
    def __init__(self):
        # Define SQL commands to create tables if they don't exist
        create_scats_table = "CREATE TABLE IF NOT EXISTS scats (scats_number INTEGER NOT NULL, " \
                             "internal_location INTEGER NOT NULL, latitude TEXT NOT NULL, longitude TEXT NOT NULL," \
                             "PRIMARY KEY (scats_number, internal_location));"

        create_data_table = "CREATE TABLE IF NOT EXISTS scats_data (id INTEGER PRIMARY KEY AUTOINCREMENT, " \
                            "scats_number INTEGER NOT NULL, internal_location INTEGER NOT NULL, date TEXT NOT NULL, " \
                            "volume INTEGER NOT NULL, FOREIGN KEY(scats_number, internal_location) " \
                            "REFERENCES scats(scats_number, internal_location));"

        # Create a connection to the SQLite database
        self.connection = sqlite3.connect("data/" + get_setting("database"))
        self.cursor = self.connection.cursor()

        # Execute the SQL commands to create tables
        self.connection.execute(create_data_table)
        self.connection.execute(create_scats_table)

        # Commit the changes to the database
        self.connection.commit()

    # Define a context manager method for using the class with 'with' statements
    def __enter__(self):
        return self

    # Define a context manager method to handle exceptions and commit or rollback changes
    def __exit__(self, ext_type, exc_value, traceback):
        self.cursor.close()
        if isinstance(exc_value, Exception):
            self.connection.rollback()
        else:
            self.connection.commit()
        self.connection.close()

    # Method to explicitly commit changes to the database
    def commit(self):
        self.connection.commit()

    # Method to insert new SCATS site information into the 'scats' table
    def insert_new_scats(self, scats_number, internal_location, latitude, longitude):
        self.cursor.execute("SELECT scats_number, internal_location FROM scats "
                            "WHERE scats_number = ? AND internal_location = ?", (scats_number, internal_location))

        data = self.cursor.fetchone()
        if data is None:
            self.connection.execute("INSERT INTO scats (scats_number, internal_location, latitude, longitude) "
                                    "VALUES (?, ?, ?, ?)", (scats_number, internal_location, latitude, longitude))

    # Method to insert SCATS data into the 'scats_data' table
    def insert_scats_data(self, scats_number, internal_location, date, volume):
        self.cursor.execute("SELECT scats_number, internal_location, date FROM scats_data "
                            "WHERE scats_number = ? AND internal_location = ? AND date = ?",
                            (scats_number, internal_location, date))

        data = self.cursor.fetchone()
        if data is None:
            self.connection.execute("INSERT INTO scats_data (scats_number, internal_location, date, volume) "
                                    "VALUES (?, ?, ?, ?)", (scats_number, internal_location, date, volume))

    # Method to retrieve SCATS volume data for a specific site and location
    def get_scats_volume(self, scats_number, internal_location):
        self.cursor.execute("SELECT volume FROM scats_data "
                            "WHERE scats_number = ? AND internal_location = ?",
                            (scats_number, internal_location))

        return np.array([item[0] for item in self.cursor.fetchall()])

    # Method to get a list of all SCATS site numbers in the 'scats' table
    def get_all_scats_numbers(self):
        self.cursor.execute("SELECT DISTINCT scats_number FROM scats")
        return [item[0] for item in self.cursor.fetchall()]

    # Method to get a list of SCATS approaches (internal locations) for a specific site
    def get_scats_approaches(self, scats_number):
        self.cursor.execute("SELECT internal_location FROM scats WHERE scats_number = ?",
                            (scats_number,))
        return [item[0] for item in self.cursor.fetchall()]
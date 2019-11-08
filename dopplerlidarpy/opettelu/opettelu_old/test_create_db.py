#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 13:38:29 2019

@author: manninan
"""

import sqlite3
from sqlite3 import Error


def create_connection(db_file):
    """Create a database connection to a SQLite database called 'db_file'

    Args:
        db_file (str): database file name

    Returns:
        conn (sqlite3.Connection): SQLite3 connection object to the database 'db_file'

    """

    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)
 
    return None


def create_table(conn, create_table_sql):
    """Create a table from the 'create_table_sql' statement

    Args:
        conn:
        create_table_sql:

    Returns:

    """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)


def create_halo_config(conn):
    """

    Args:
        conn ():

    Returns:

    """

    halo_config_query = """ CREATE TABLE IF NOT EXISTS halo_config (
                                site text PRIMARY KEY,
                                parameters_valid_from_including datetime,
                                lat integer NOT NULL,
                                lon integer NOT NULL,
                                altitude_instrument_level_m_asl integer NOT NULL,
                                altitude_ground_level_m_asl integer NOT NULL,
                                ); """

    create_table(conn, halo_config_query)


def main(db_file):

    halo_config = """ CREATE TABLE IF NOT EXISTS halo_config (
                   site text PRIMARY KEY,
                   parameters_valid_from_including datetim
                   lat integer NOT NULL,
                   lon integer NOT NULL,
                   altitude_instrument_level_m_asl integer NOT NULL
                   altitude_ground_level_m_asl integer NOT NULL
                   ); """
 
    Products = """CREATE TABLE IF NOT EXISTS Products (
                                    id integer
                                    name text PRIMARY KEY,
                                    FOREIGN KEY (site) REFERENCES halo_config (unit_id)
                                );"""
 
    # create a database connection
    conn = create_connection(db_file)
    if conn is not None:
        # create projects table
        create_table(conn, measurements)
        # create tasks table
        create_table(conn, products)
    else:
        print("Error! cannot create the database connection.")


if __name__ == '__main__':
    main()

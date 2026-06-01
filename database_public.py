import psycopg2
import time

class DataBase:
    def __init__(self, db:str, table_structure:list[str]):
        self.conn = psycopg2.connect(
            dbname=db,
            user="user_of_your_database",
            password=r"your_password",
            host="localhost",
            port=0
        )

        self.cur = self.conn.cursor()

        self.conn.commit()

        self.table_structure = ""

        self.change_table(table_structure)

    def get_data(self, command:str):
        self.cur.execute(command)
        return self.cur.fetchall()
    
    def insert_data(self, values:list[tuple[str]], params:list[str], table:str):
        table_params = ""
        for i in range(len(params)):
            table_params += params[i]

            if i < len(params) - 1:
                table_params += ", "

        table_values = ""
        for i in range(len(values)):
            table_values += str(tuple(values[i]))

            if i < len(values) - 1:
                table_values += ", "

        self.cur.execute(f"INSERT INTO {table} ({table_params}) VALUES {table_values}")
        self.conn.commit()

    def delete_data(self, table:str, specifications:str):
        if specifications == "":
            self.cur.execute(f"DELETE FROM {table}")
        else:
            self.cur.execute(f"DELETE FROM {table} {specifications}")

        self.conn.commit()

    def change_table(self, table_structure:list[str]):
        table_params = ""
        for i in range(1, len(table_structure)):
            table_params += table_structure[i]

            if i < len(table_structure) - 1:
                table_params += ", "

        self.table_structure = f"{table_structure[0]} ({table_params})"
        self.cur.execute(f"""CREATE TABLE IF NOT EXISTS {self.table_structure}""")
        self.conn.commit()

    def close_connection(self):
        self.cur.close()
        self.conn.close()
import os
from google.cloud import bigquery


class BigQuery(object):
    def __new__(cls, *args, **kwargs):
        instance = object.__new__(cls)
        return instance

    def __init__(self, cred_file_path):
        self.cred_file_path = cred_file_path
        self.init_credentials()
        self.client = bigquery.Client()
        self.dataset = None
        self.dataset_ref = None

    def init_credentials(self):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.cred_file_path

    def hello(self):
        print("hello")

    def init_dataset(self, dataset_name, project):
        self.dataset_ref = self.client.dataset(dataset_name, project=project)
        self.dataset = self.client.get_dataset(self.dataset_ref)

    def tables(self):
        return self.client.list_tables(self.dataset)

    def get_table(self, name):
        table_ref = self.dataset_ref.table(name)
        return self.client.get_table(table_ref)

    def list_rows(self, table, **kwargs):
        return self.client.list_rows(
            table, max_results=kwargs["max_results"]
        ).to_dataframe()

    def run_query(self, query, **kwargs):
        run_config = bigquery.QueryJobConfig(**kwargs)
        return self.client.query(query, run_config)

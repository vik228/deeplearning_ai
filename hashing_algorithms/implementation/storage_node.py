import requests


class StorageNode(object):
    def __init__(self, name, host):
        self.name = name
        self.host = host

    def fetch_file(self, path):
        return requests.get(f'https://{self.host}:1234/{path}').text

    def put_file(self, path):
        with open(path, 'r') as fp:
            content = fp.read()
            return requests.post(f'https://{self.host}:1231/{path}', body=content).text

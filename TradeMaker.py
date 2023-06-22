import requests

class TradeMaker:

    def __init__(self):
        self.token = 'mUoTZU1LAk3yMgOIyLSfJwdNnwAr'

    def checkStatus(self):
        response = requests.get('https://api.tradier.com/v1/user/profile',
        params={},
        headers={'Authorization': 'Bearer '+self.token, 'Accept': 'application/json'}
        )
        json_response = response.json()
        return response.status_code == 200

    def getAccountInfo(self):
        response = requests.get('https://api.tradier.com/v1/user/profile',
        params={},
        headers={'Authorization': 'Bearer '+self.token, 'Accept': 'application/json'}
        )
        json_response = response.json()
        print(response.status_code)
        print(json_response)


    def getPositions(self):
        response = requests.get('https://api.tradier.com/v1/accounts/{account_id}/positions',
        params={},
        headers={'Authorization': 'Bearer <TOKEN>', 'Accept': 'application/json'}
        )
        json_response = response.json()
        print(response.status_code)
        return json_response

    def liquidate(self):
        #get portfolio, sell all of it.
        

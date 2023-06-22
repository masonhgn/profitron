import requests

class TradeMaker:
    def __init__(self):
        self.token = 'mUoTZU1LAk3yMgOIyLSfJwdNnwAr'
        self.account_number = '6YA32927'
    def check_status(self):
        response = requests.get('https://api.tradier.com/v1/user/profile',
        params={},
        headers={'Authorization': 'Bearer '+self.token, 'Accept': 'application/json'}
        )
        json_response = response.json()
        return response.status_code == 200

    def get_account_info(self):
        response = requests.get('https://api.tradier.com/v1/user/profile',
        params={},
        headers={'Authorization': 'Bearer '+self.token, 'Accept': 'application/json'}
        )
        json_response = response.json()
        print(response.status_code)
        print(json_response)

    def get_balance(self):
        response = requests.get('https://api.tradier.com/v1/accounts/'+self.account_number+'/balances',
        params={},
        headers={'Authorization': 'Bearer '+self.token, 'Accept': 'application/json'}
        )
        if response.status_code != 200:
            print('api call during get_balance function call unsuccessful: ' + response.status_code)
            return None
        json_response = response.json()
        return json_response

    def get_cash_available(self):
        balance = self.getBalance()
        print(balance['cash_available'])

    def get_positions(self):
        response = requests.get('https://api.tradier.com/v1/accounts/'+self.account_number+'/positions',
        params={},
        headers={'Authorization': 'Bearer <TOKEN>', 'Accept': 'application/json'}
        )
        json_response = response.json()
        print(response.status_code)
        return json_response

    def liquidate(self):
        pass
        #get portfolio, sell all of it.
        

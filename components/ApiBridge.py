import requests

class ApiBridge:
    def __init__(self, production):
        self.reinit(production)

    def reinit(self,production):
        """toggles between sandbox mode and live mode"""
        self.production = production

        self.token = 'mUoTZU1LAk3yMgOIyLSfJwdNnwAr' if self.production else 'LUiHZ2fsw2f8lfptk0UTmLH5BnrX'
        self.url = 'https://api.tradier.com' if self.production else 'https://sandbox.tradier.com'
        self.account_number = '6YA32927' if self.production else 'VA43237255'
        self.headers = headers={'Authorization': 'Bearer '+self.token, 'Accept': 'application/json'}




    def check_status(self):
        """gets account status"""
        return requests.get(
            f'{self.url}/v1/user/profile',
            params={},
            headers=self.headers
        ).json()



    def get_balance_metrics(self):
        """returns all balance metrics of account"""
        return requests.get(
            f'{self.url}/v1/accounts/'+self.account_number+'/balances',
            params={},
            headers=self.headers
        ).json()




    def get_positions(self):
        """gets all current positions"""
        return requests.get(
            f'{self.url}/v1/accounts/'+self.account_number+'/positions',
            params={},
            headers=self.headers
        ).json()



    def liquidate(self):
        """liquidates entire portfolio"""
        pass
        #get portfolio, sell all of it.




    def equity_limit_order(self, symbol, side, quantity, duration, price):
        """place a limit equity order"""
        response = requests.get(
            f'{self.url}/v1/accounts/'+self.account_number+'/orders',
            params={
                'class':'equity',
                'symbol':symbol,
                'side':side,
                'quantity':str(quantity),
                'type':'limit',
                'duration':duration,
                'price': str(price)
            },
            headers=self.headers
        ).json()

        return response




    def equity_market_order(self, symbol, side, quantity, duration):
        """place a market equity order"""
        response = requests.get(
            f'{self.url}/v1/accounts/'+self.account_number+'/orders',
            params={
                'class':'equity',
                'symbol':symbol,
                'side':side,
                'quantity':quantity,
                'type':'market',
                'duration':duration
            },
            headers=self.headers
        ).json()

        return response

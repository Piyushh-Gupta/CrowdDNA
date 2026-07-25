class Paginator:
    def __init__(self, transport, path, params=None):
        self.transport = transport
        self.path = path
        self.params = params or {}
        
    def __iter__(self):
        cursor = self.params.get('cursor')
        page = self.params.get('page', 1)
        
        while True:
            if cursor:
                self.params['cursor'] = cursor
            else:
                self.params['page'] = page
                
            resp = self.transport.send("GET", self.path, params=self.params).json()
            items = resp.get('items', [])
            
            if not items:
                break
                
            for item in items:
                yield item
                
            if 'next_cursor' in resp:
                if resp['next_cursor']:
                    cursor = resp['next_cursor']
                    # Clear page if switching to cursor
                    self.params.pop('page', None)
                else:
                    break
            elif 'page' in self.params:
                page += 1
            else:
                break

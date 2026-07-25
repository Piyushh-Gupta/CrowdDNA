class AdminPermissions:
    def has_access(self, operator, action):
        return operator == "admin"

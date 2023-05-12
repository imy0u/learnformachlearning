class Screen(object):
    @property
    def width(self):
        return self._width
    
    @property
    def height(self):
        return self._height
    
    @property
    def resolution(self):
        return self._width*self._height
    
    @width.setter
    def width(self,value):
        self._width=value

    @height.setter
    def height(self,value1):
        self._height=value1

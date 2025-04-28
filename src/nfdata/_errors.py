class RoutingError(Exception):
    """Exception raised for errors while routing waterbodies."""


def whitebox_callback(msg):
    """
    Callback function to catch Whitebox errors
    """
    if 'Error' in msg:
        raise RoutingError(f'Whitebox raised an error: '
                           f'{msg}')

import sys
import pytest
import torch as th
import syft as sy
from syft.core.node.common.service.auth import AuthorizationException

sy.LOG_FILE = "syft_do.log"
sy.logger.remove()
_ = sy.logger.add(sys.stdout, level="DEBUG")


# We need this for the DEMO purpose because at the training time
# we want to see the loss and for doing that (in a real world scenario)
# we will have to do a request and then to get it approved by the data owner
# Since training might generate a lot of request and we know the VM is locally
# we kind of approve those requests locally
def get_permission(obj):
    remote_obj = alice.store[obj.id_at_location]
    remote_obj.read_permissions[alice_client.verify_key] = obj.id_at_location


alice = sy.VirtualMachine(name="alice")
alice_client = alice.get_root_client()

remote_python = alice_client.syft.lib.python



import sys
import pytest
import torch as th
import syft as sy
from syft.core.node.common.service.auth import AuthorizationException

sy.LOG_FILE = "syft_do.log"
sy.logger.remove()
_ = sy.logger.add(sys.stdout, level="DEBUG")

x = th.tensor([1,2,3,4,5])
print(x)


somedevice = sy.Device()
# print(somedevice.name, somedevice.id, somedevice.address)
# print(somedevice.address.vm, somedevice.address.device, somedevice.address.domain, somedevice.address.network)
# print(somedevice.address.target_id)
# print(somedevice.address.target_id.pprint)
# print(somedevice.address.pprint)


bob_device = sy.Device(name="Bob's iPhone")
assert bob_device.name == "Bob's iPhone"

bob_device_client = bob_device.get_client()

msg = sy.ReprMessage(address=bob_device_client.address)
print(msg.pprint)
print(bob_device_client.address.pprint)
assert msg.address == bob_device_client.address

with pytest.raises(AuthorizationException):
    bob_device_client.send_immediate_msg_without_reply(
        msg=sy.ReprMessage(address=bob_device_client.address)
    )


print(bob_device_client.keys)
print(bob_device.keys)

assert bob_device_client.verify_key != bob_device.root_verify_key


bob_device_client = bob_device.get_root_client()

# repr_service.py
# class ReprService(ImmediateNodeServiceWithoutReply):
#     @staticmethod
#     @service_auth(root_only=True)


bob_vm = sy.VirtualMachine(name="Bob's VM")
bob_vm_client = bob_vm.get_root_client()

bob_device_client.register(client=bob_vm_client)

bob_domain = sy.Domain(name="Bob's Domain")
bob_domain_client = bob_domain.get_root_client()

print(bob_domain.address.pprint)
bob_domain_client.register(client=bob_device_client)





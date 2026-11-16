import ipaddress
import unittest
from unittest.mock import patch

from plaud_workflow.web import serve

# RFC 6598 shared address space, the range Tailscale assigns from.
SHARED = ipaddress.ip_network('100.64.0.0/10')
TAILNET_HOST = str(SHARED[0x10203])                                  # an arbitrary address inside it
OUTSIDE_SHARED = str(SHARED.network_address + (136 << 16) + 0x0101)  # same first octet, outside the /10


class ServingBeyondLoopback(unittest.TestCase):
    """The owner asked for the site on Tailscale. Other devices are on that tailnet too,
    so widening the address must not also drop the session."""

    def base(self,**kw):
        c={'host':'127.0.0.1','port':8767,'root':'/tmp/x','database':'/tmp/x.sqlite',
           'base_url':'http://127.0.0.1:8767','require_session':False}
        c.update(kw);return c

    def test_the_lan_is_still_refused(self):
        for host in ('0.0.0.0','192.0.2.51',OUTSIDE_SHARED,'10.0.0.4'):
            with self.subTest(host=host), self.assertRaisesRegex(ValueError,'loopback or'):
                serve(self.base(host=host,require_session=True))

    def test_a_tailnet_address_without_a_session_is_refused(self):
        with self.assertRaisesRegex(ValueError,'require_session must be on'):
            serve(self.base(host=TAILNET_HOST,require_session=False))

    def test_loopback_still_needs_no_session(self):
        with patch('plaud_workflow.web.Application'),patch('plaud_workflow.web.ThreadingHTTPServer') as srv:
            serve(self.base())
        self.assertTrue(srv.called)

    def test_a_tailnet_address_with_a_session_serves(self):
        with patch('plaud_workflow.web.Application'),patch('plaud_workflow.web.ThreadingHTTPServer') as srv:
            serve(self.base(host=TAILNET_HOST,require_session=True,
                            base_url=f'http://{TAILNET_HOST}:8767'))
        self.assertEqual(srv.call_args[0][0],(TAILNET_HOST,8767))


if __name__=='__main__':unittest.main()


class LauncherTicket(unittest.TestCase):
    """A ticket handed to a person on another device has to outlive the handover."""

    def app(self,**kw):
        import tempfile
        from plaud_workflow.web import Application
        d=tempfile.mkdtemp()
        c={'root':d,'database':d+'/s.sqlite','base_url':'http://127.0.0.1:8767',
           'host':'127.0.0.1','port':8767}
        c.update(kw);return Application(c)

    def test_the_window_defaults_to_ten_minutes_not_one(self):
        import time as t
        a=self.app();a.ticket(a.key)
        self.assertGreater(min(a.tickets.values())-t.time(),300)

    def test_a_wrong_key_mints_nothing(self):
        a=self.app()
        with self.assertRaises(PermissionError):a.ticket('not-the-key')
        self.assertEqual(a.tickets,{})

    def test_a_ticket_is_single_use(self):
        a=self.app();t=a.ticket(a.key)
        a.exchange(t)
        with self.assertRaises(PermissionError):a.exchange(t)

    def test_an_expired_ticket_is_refused(self):
        a=self.app(ticket_seconds=-1);t=a.ticket(a.key)
        with self.assertRaisesRegex(PermissionError,'expired'):a.exchange(t)

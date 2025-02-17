#!/usr/bin/env python

##################################################################### imports

import os
import sys

if sys.version_info.major == 2:
    import gtk
else:
    import pgi
    pgi.install_as_gi()
    from gi.repository import Gtk as gtk

import DeFiNe
import DeFiNe.calc

##################################################################### run

class DeFiNe_class:

    def __init__(self):
        self.builder=gtk.Builder()
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        self.builder.add_from_file('../data/gui.glade')
        self.window=self.builder.get_object('window1')
        self.builder.connect_signals(self)
        self.window.show()

    def on_window1_destroy(self,object):
        print('Closing DeFiNe.')
        gtk.main_quit()

    def on_button1_clicked(self,object):
        print('Running DeFiNe.')
        inp=str(self.builder.get_object('filechooserbutton1').get_filename()).replace('\\','/')
        print(inp)
        sampling=self.builder.get_object('combobox1').get_active()
        overlap=1-self.builder.get_object('combobox2').get_active()
        quality=self.builder.get_object('combobox3').get_active()
        objective=self.builder.get_object('combobox4').get_active()
        angle=float(self.builder.get_object('spinbutton1').get_value())
        DeFiNe.calc.calc(self,gtk,inp,sampling,overlap,quality,objective,angle)
        print('Completing DeFiNe.')
        return None


if __name__ == '__main__':
    app=DeFiNe_class()
    gtk.main()






















"""Irregular verbs: the facts of grammar the rules cannot derive.

Every entry is about a lexeme and its stems, never about a verse's token:
the second aorists, suppletive stems (ὁράω ... εἶδον, ὄψομαι, ὤφθην; φέρω ...
ἤνεγκα, οἴσω; λέγω ... εἶπον, ἐρῶ, ἐρρέθην; ἔρχομαι ... ἦλθον, ἐλεύσομαι,
ἐλήλυθα), irregular futures, perfects and passives, and the athematic
(-μι) verbs, whose presents and root aorists are written out. A compound
inherits its simplex's entry (ἀπο-δίδωμι, κατα-λαμβάνω) through the
analyser's preverb splitting, so only simplex verbs are listed.

PRINCIPAL_PARTS maps a simplex lemma to stems, written
`system:stem[/augmented,augmented][*flags]`:
  systems  fut, aor1, aor2, aorp, aorp2, perf, perfm, pres (an extra
           present stem: θνήσκω beside θνῄσκω), impf (an irregular
           imperfect stem: ἑώρων), fut and aor1 for liquid stems take *c
           (contracted future κρινῶ) where needed
  flags    2  TAGNT writes the system's codes with a leading 2 (second
              aorist, second perfect)
           c  contracted (ε) future: βαλῶ, ἐρῶ, πεσοῦμαι; on a present, a
              Koine contract present beside the athematic one (ἵημι's
              -ιῶ, -ιεῖ, -ιοῦσι: συνιοῦσι)
           w  a weaker alternative
The augmented spellings default to the rule's (`morph_phon.augment`).

TABLES hold written-out paradigms, one line per code: `PaI-3S δίδωσι
δίδωσιν`. The code is TAGNT's tense (with 2), a slot letter for the voice
(a active, m present/perfect middle-passive, f future middle, s aorist
middle, p aorist passive, q future passive), the mood, and person or
case-number-gender. A form ending in ! keeps its accent in a compound
(contracted ἱστᾶσι); other finite forms are re-accented there (δίδωσι,
ἀποδίδωσι). `@ptc` lines build a participle's 24 forms from a body and a
type (see `_PARTICIPLE_TYPES`).
"""

from __future__ import annotations

import re

from lxx_tagging import morph_phon as ph
from lxx_tagging.morph_verbs import Stem, _ptc_third, _second_declension

PRINCIPAL_PARTS: dict[str, str] = {
    # Added for Septuagint forms the rules missed (round 2): the augment of
    # εὐ- verbs on their second vowel as the Septuagint prints it
    # (εὐώδωσεν, εὐηρέστησε), ἐκκλησιάζω's double augment (ἐξεκκλησίασε),
    # and the stems of ἕψω, πέσσω, σπένδω, ἁλίσκομαι.
    "εὐοδόω": "fut:εὐοδωσ aor1:εὐοδωσ/εὐωδωσ,εὐοδωσ aorp:εὐοδωθ/εὐωδωθ,εὐοδωθ perf:εὐοδωκ,εὐωδωκ",
    "εὐαρεστέω": "fut:εὐαρεστησ aor1:εὐαρεστησ/εὐηρεστησ perf:εὐηρεστηκ",
    "ἐκκλησιάζω": "fut:ἐκκλησιασ aor1:ἐκκλησιασ/ἐξεκκλησιασ aorp:ἐκκλησιασθ/ἐξεκκλησιασθ",
    "ἕψω": "fut:ἑψησ aor1:ἑψησ/ἡψησ aorp:ἑψηθ/ἡψηθ perfm:ἡψη",
    "πέσσω": "fut:πεψ aor1:πεψ/ἐπεψ aorp:πεφθ/ἐπεφθ perfm:πεπεμ",
    "σπένδω": "fut:σπεισ aor1:σπεισ/ἐσπεισ aorp:σπεισθ/ἐσπεισθ",
    "ἁλίσκομαι": "fut:ἁλωσ perf:ἑαλωκ,ἡλωκ",
    "οἰκτείρω": "fut:οἰκτειρησ aor1:οἰκτειρησ/ᾠκτειρησ fut:οἰκτιρησ*w aor1:οἰκτιρησ/ᾠκτιρησ*w",
    "ἄγω": "aor2:ἀγαγ/ἠγαγ*2 fut:ἀξ aor1:ἀξ/ἠξ*w aorp:ἀχθ/ἠχθ perfm:ἠγ perf:ἀγηοχ*w",
    "αἱρέω": "fut:αἱρησ fut:ἑλ*c aor2:ἑλ/εἱλ*2 aorp:αἱρεθ/ᾑρεθ perf:ᾑρηκ perfm:ᾑρη",
    "αἴρω": "fut:ἀρ*c aor1:ἀρ/ἠρ aorp:ἀρθ/ἠρθ perf:ἠρκ perfm:ἠρ",
    "αἰσθάνομαι": "aor2:αἰσθ/ᾐσθ*2 fut:αἰσθησ",
    "ἁμαρτάνω": "aor2:ἁμαρτ/ἡμαρτ*2 aor1:ἁμαρτησ/ἡμαρτησ fut:ἁμαρτησ perf:ἡμαρτηκ aorp:ἁμαρτηθ/ἡμαρτηθ",
    "ἀκούω": "fut:ἀκουσ aor1:ἀκουσ/ἠκουσ aorp:ἀκουσθ/ἠκουσθ perf:ἀκηκο*2 perfm:ἠκουσ",
    "ἀναλίσκω": "fut:ἀναλωσ aor1:ἀναλωσ/ἀνηλωσ aorp:ἀναλωθ/ἀνηλωθ perf:ἀνηλωκ perfm:ἀνηλω",
    "ἀνοίγω": ("fut:ἀνοιξ aor1:ἀνοιξ/ἠνοιξ,ἀνεῳξ,ἠνεῳξ aorp:ἀνοιχθ/ἠνοιχθ,ἀνεῳχθ,ἠνεῳχθ "
               "aorp2:ἀνοιγ/ἠνοιγ,ἀνεῳγ*2 perf:ἀνεῳγ*2 perfm:ἀνεῳγ perfm:ἠνεῳγ impf:ἠνοιγ,ἀνεῳγ"),
    "ἀρέσκω": "fut:ἀρεσ aor1:ἀρεσ/ἠρεσ",
    "αὐξάνω": "fut:αὐξησ aor1:αὐξησ/ηὐξησ aorp:αὐξηθ/ηὐξηθ",
    "βαίνω": "fut:βησ perf:βεβηκ",
    "βάλλω": "fut:βαλ*c aor2:βαλ/ἐβαλ*2 aorp:βληθ/ἐβληθ perf:βεβληκ perfm:βεβλη",
    "βάπτω": "aorp:βαφ/ἐβαφ aorp2:βαφ/ἐβαφ*2",
    "βιόω": "aor2:βι/ἐβι*2",
    "βλαστάνω": "aor2:βλαστ/ἐβλαστ*2 aor1:βλαστησ/ἐβλαστησ",
    "βούλομαι": "fut:βουλησ aorp:βουληθ/ἐβουληθ,ἠβουληθ impf:ἐβουλ,ἠβουλ perfm:βεβουλη",
    "γαμέω": "aor1:γημ/ἐγημ aor1:γαμησ/ἐγαμησ",
    "γηράσκω": "fut:γηρασ aor1:γηρασ/ἐγηρασ",
    "γίνομαι": "aor2:γεν/ἐγεν*2 aorp:γενηθ/ἐγενηθ fut:γενησ perf:γεγον*2 perfm:γεγενη",
    "γίγνομαι": "aor2:γεν/ἐγεν*2 aorp:γενηθ/ἐγενηθ fut:γενησ perf:γεγον*2 perfm:γεγενη",
    "γινώσκω": "fut:γνωσ aorp:γνωσθ/ἐγνωσθ perf:ἐγνωκ perfm:ἐγνωσ",
    "γιγνώσκω": "fut:γνωσ aorp:γνωσθ/ἐγνωσθ perf:ἐγνωκ perfm:ἐγνωσ",
    "γράφω": "aorp2:γραφ/ἐγραφ*2 perf:γεγραφ*2 perfm:γεγραφ",
    "δάκνω": "aor2:δακ/ἐδακ*2 fut:δηξ aorp:δηχθ/ἐδηχθ",
    "δείκνυμι": "fut:δειξ aor1:δειξ/ἐδειξ aorp:δειχθ/ἐδειχθ perf:δεδειχ perfm:δεδεικ",
    "δέομαι": "fut:δεησ aorp:δεηθ/ἐδεηθ",
    "δέω": "fut:δησ aor1:δησ/ἐδησ aorp:δεθ/ἐδεθ perf:δεδεκ perfm:δεδε",
    "διδάσκω": "fut:διδαξ aor1:διδαξ/ἐδιδαξ aorp:διδαχθ/ἐδιδαχθ perf:δεδιδαχ perfm:δεδιδαγ",
    "διώκω": "fut:διωξ aor1:διωξ/ἐδιωξ aorp:διωχθ/ἐδιωχθ perfm:δεδιωκ",
    "δοκέω": "fut:δοξ aor1:δοξ/ἐδοξ perfm:δεδογ",
    "δύναμαι": "fut:δυνησ aorp:δυνηθ/ἠδυνηθ,ἐδυνηθ aorp:δυνασθ/ἠδυνασθ impf:ἐδυνα,ἠδυνα",
    "δύνω": "fut:δυσ aor1:δυσ/ἐδυσ perf:δεδυκ",
    "δύω": "fut:δυσ aor1:δυσ/ἐδυσ perf:δεδυκ",
    "ἐάω": "fut:ἐασ aor1:ἐασ/εἰασ impf:εἰα",
    "ἐγείρω": "fut:ἐγερ*c aor1:ἐγειρ/ἠγειρ aorp:ἐγερθ/ἠγερθ perfm:ἐγηγερ perf:ἐγηγερκ",
    "ἐθίζω": "impf:εἰθιζ perfm:εἰθισ",
    "εἶδον": "aor2:ἰδ/εἰδ,ἰδ*2",
    "εἴδω": "aor2:ἰδ/εἰδ,ἰδ*2 fut:εἰδησ",
    "εἶπον": "aor2:εἰπ/εἰπ*2 aor1:εἰπ/εἰπ*2w",
    "ἐλαύνω": "fut:ἐλασ aor1:ἐλασ/ἠλασ perf:ἐληλακ perfm:ἐληλα aorp:ἐλαθ/ἠλαθ",
    "ἕλκω": "fut:ἑλκυσ aor1:ἑλκυσ/εἱλκυσ aorp:ἑλκυσθ/εἱλκυσθ impf:εἱλκ",
    "ἐλπίζω": "fut:ἐλπι*c",
    "ἐργάζομαι": "impf:εἰργαζ,ἠργαζ aor1:ἐργασ/εἰργασ,ἠργασ aorp:ἐργασθ/εἰργασθ,ἠργασθ perfm:εἰργασ",
    "ἔρχομαι": "aor2:ἐλθ/ἠλθ*2 fut:ἐλευσ perf:ἐληλυθ*2",
    "ἐρωτάω": "fut:ἐρωτησ aor1:ἐρωτησ/ἠρωτησ",
    "ἐσθίω": "aor2:φαγ/ἐφαγ*2 fut:φαγ fut:ἐδ aorp:βρωθ/ἐβρωθ perf:βεβρωκ perfm:βεβρω",
    "ἔσθω": "aor2:φαγ/ἐφαγ*2 fut:φαγ",
    "εὑρίσκω": ("aor2:εὑρ/εὑρ,ηὑρ*2 aor1:εὑρ/εὑρ,ηὑρ*2w fut:εὑρησ aorp:εὑρεθ/εὑρεθ,ηὑρεθ "
                "perf:εὑρηκ perf:ηὑρηκ perfm:εὑρη"),
    "ἔχω": "aor2:σχ/ἐσχ*2 fut:ἑξ fut:σχησ impf:εἰχ perf:ἐσχηκ aorp:σχεθ/ἐσχεθ",
    "ζάω": "fut:ζησ aor1:ζησ/ἐζησ",
    "θάπτω": "fut:θαψ aor1:θαψ/ἐθαψ aorp2:ταφ/ἐταφ*2 perfm:τεθαφ",
    "θέλω": "fut:θελησ aor1:θελησ/ἠθελησ impf:ἠθελ aorp:θεληθ/ἠθεληθ",
    "θνῄσκω": "aor2:θαν/ἐθαν*2 fut:θαν*c perf:τεθνηκ pres:θνησκ",
    "θνήσκω": "aor2:θαν/ἐθαν*2 fut:θαν*c perf:τεθνηκ pres:θνῃσκ",
    "θιγγάνω": "aor2:θιγ/ἐθιγ*2",
    "ἱκνέομαι": "aor2:ἱκ/ἱκ*2 fut:ἱξ",
    "καίω": "fut:καυσ aor1:καυσ/ἐκαυσ aorp:καυθ/ἐκαυθ aorp2:καη/ἐκαη perfm:κεκαυ perf:κεκαυκ",
    "καλέω": "fut:καλεσ fut:καλε*c aor1:καλεσ/ἐκαλεσ aorp:κληθ/ἐκληθ perf:κεκληκ perfm:κεκλη",
    "κάμνω": "aor2:καμ/ἐκαμ*2 perf:κεκμηκ",
    "κερδαίνω": "fut:κερδησ aor1:κερδησ/ἐκερδησ aor1:κερδαν/ἐκερδαν",
    "κλαίω": "fut:κλαυσ aor1:κλαυσ/ἐκλαυσ",
    "κλείω": "fut:κλεισ aor1:κλεισ/ἐκλεισ aorp:κλεισθ/ἐκλεισθ perfm:κεκλεισ",
    "κλίνω": "fut:κλιν*c aor1:κλιν/ἐκλιν aorp:κλιθ/ἐκλιθ perf:κεκλικ perfm:κεκλι",
    "κόπτω": "aorp2:κοπ/ἐκοπ*2 perfm:κεκοπ",
    "κράζω": "fut:κραξ fut:κεκραξ aor1:κραξ/ἐκραξ aor2:κραγ/ἐκραγ*2 perf:κεκραγ*2",
    "μάχομαι": "fut:μαχε*c fut:μαχεσ aor1:μαχεσ/ἐμαχεσ aorp:μαχεσθ/ἐμαχεσθ",
    "ὠθέω": "fut:ὠσ aor1:ὠσ/ἐωσ,ὠσ aorp:ὠσθ/ἐωσθ,ὠσθ perfm:ἐωσ",
    "τήκω": "fut:τηξ aor1:τηξ/ἐτηξ aorp2:τακ/ἐτακ*2 aorp:τηχθ/ἐτηχθ perf:τετηκ*2",
    "σκέπτομαι": "fut:σκεψ aor1:σκεψ/ἐσκεψ aorp2:σκεπ/ἐσκεπ*2",
    "κλέπτω": "fut:κλεψ aor1:κλεψ/ἐκλεψ aorp2:κλαπ/ἐκλαπ*2 perf:κεκλοφ*2",
    "διδράσκω": "fut:δρασ",
    "βόσκω": "fut:βοσκησ aor1:βοσκησ/ἐβοσκησ aorp:βοσκηθ/ἐβοσκηθ",
    "κρεμάννυμι": "fut:κρεμασ aor1:κρεμασ/ἐκρεμασ aorp:κρεμασθ/ἐκρεμασθ",
    "κρίνω": "fut:κριν*c aor1:κριν/ἐκριν aorp:κριθ/ἐκριθ perf:κεκρικ perfm:κεκρι",
    "κρύπτω": "aorp2:κρυβ/ἐκρυβ*2 aorp:κρυφθ/ἐκρυφθ perfm:κεκρυπ",
    "κτείνω": "fut:κτεν*c aor1:κτειν/ἐκτειν aorp:κτανθ/ἐκτανθ aorp2:κταν/ἐκταν*2 pres:κτενν",
    "λαγχάνω": "aor2:λαχ/ἐλαχ*2",
    "λαμβάνω": ("aor2:λαβ/ἐλαβ*2 fut:λημψ fut:ληψ aorp:λημφθ/ἐλημφθ aorp:ληφθ/ἐληφθ perf:εἰληφ*2 "
                "perfm:εἰλημπ perfm:εἰληπ"),
    "λανθάνω": "aor2:λαθ/ἐλαθ*2 fut:λησ perfm:λελησ",
    "λέγω": "aor2:εἰπ/εἰπ*2 fut:ἐρ*c aorp:ῥηθ/ἐρρηθ,ἐρρεθ fut:λεξ aor1:λεξ/ἐλεξ perf:εἰρηκ perfm:εἰρη",
    "λείπω": "aor2:λιπ/ἐλιπ*2 fut:λειψ aorp:λειφθ/ἐλειφθ perf:λελοιπ*2 perfm:λελειπ",
    "μανθάνω": "aor2:μαθ/ἐμαθ*2 fut:μαθησ perf:μεμαθηκ",
    "μέλλω": "fut:μελλησ impf:ἐμελλ,ἠμελλ",
    "μένω": "fut:μεν*c aor1:μειν/ἐμειν perf:μεμενηκ",
    "μιαίνω": "fut:μιαν*c aor1:μιαν/ἐμιαν aorp:μιανθ/ἐμιανθ perfm:μεμιαν",
    "μιμνῄσκω": "fut:μνησ aor1:μνησ/ἐμνησ aorp:μνησθ/ἐμνησθ perfm:μεμνη",
    "μιμνήσκω": "fut:μνησ aor1:μνησ/ἐμνησ aorp:μνησθ/ἐμνησθ perfm:μεμνη",
    "ὄλλυμι": "fut:ὀλ*c fut:ὀλεσ aor1:ὀλεσ/ὠλεσ aor2:ὀλ/ὠλ*2 perf:ὀλωλ*2 perf:ὀλωλεκ",
    "ὄμνυμι": "fut:ὀμ*c aor1:ὀμοσ/ὠμοσ perf:ὀμωμοκ",
    "ὁράω": ("aor2:ἰδ/εἰδ,ἰδ*2 fut:ὀψ aorp:ὀφθ/ὠφθ perf:ἑωρακ perf:ἑορακ perfm:ἑωρα impf:ἑωρα"),
    "ὀφείλω": "aor2:ὀφελ/ὠφελ*2 impf:ὠφειλ",
    "πάσχω": "aor2:παθ/ἐπαθ*2 fut:πεισ perf:πεπονθ*2",
    "πείθω": "fut:πεισ aor1:πεισ/ἐπεισ aorp:πεισθ/ἐπεισθ perf:πεποιθ*2 perfm:πεπεισ",
    "πίνω": "aor2:πι/ἐπι*2 fut:πι aorp:ποθ/ἐποθ perf:πεπωκ perfm:πεπο",
    "πίπτω": "aor2:πεσ/ἐπεσ*2 fut:πεσ*c perf:πεπτωκ",
    "πλέω": "fut:πλευσ aor1:πλευσ/ἐπλευσ",
    "πλήσσω": "aorp2:πληγ/ἐπληγ*2 aorp2:πλαγ/ἐπλαγ*2 perfm:πεπληγ perf:πεπληγ*2",
    "πνέω": "fut:πνευσ aor1:πνευσ/ἐπνευσ",
    "πράσσω": "fut:πραξ aor1:πραξ/ἐπραξ aorp:πραχθ/ἐπραχθ perf:πεπραχ*2 perfm:πεπραγ",
    "πυνθάνομαι": "aor2:πυθ/ἐπυθ*2 fut:πευσ",
    "ῥέω": "fut:ῥευσ aorp2:ῥυ/ἐρρυ*2",
    "ῥήγνυμι": "fut:ῥηξ aor1:ῥηξ/ἐρρηξ aorp2:ῥαγ/ἐρραγ*2 perf:ἐρρωγ*2",
    "ῥίπτω": "aorp2:ῥιφ/ἐρριφ*2 aorp:ῥιφθ/ἐρριφθ perfm:ἐρριπ",
    "σπείρω": "fut:σπερ*c aor1:σπειρ/ἐσπειρ aorp2:σπαρ/ἐσπαρ*2 perfm:ἐσπαρ",
    "στέλλω": "fut:στελ*c aor1:στειλ/ἐστειλ aorp2:σταλ/ἐσταλ*2 perf:ἐσταλκ perfm:ἐσταλ",
    "στρέφω": "aorp2:στραφ/ἐστραφ*2 aorp:στρεφθ/ἐστρεφθ perfm:ἐστραπ",
    "σῴζω": "fut:σωσ aor1:σωσ/ἐσωσ aorp:σωθ/ἐσωθ perf:σεσωκ perfm:σεσω perfm:σεσωσ",
    "σώζω": "fut:σωσ aor1:σωσ/ἐσωσ aorp:σωθ/ἐσωθ perf:σεσωκ perfm:σεσω perfm:σεσωσ",
    "τάσσω": "fut:ταξ aor1:ταξ/ἐταξ aorp2:ταγ/ἐταγ*2 aorp:ταχθ/ἐταχθ perf:τεταχ perfm:τεταγ",
    "τείνω": "fut:τεν*c aor1:τειν/ἐτειν aorp:ταθ/ἐταθ perfm:τετα",
    "τελέω": "fut:τελεσ aor1:τελεσ/ἐτελεσ aorp:τελεσθ/ἐτελεσθ perf:τετελεκ perfm:τετελεσ",
    "τέμνω": "aor2:τεμ/ἐτεμ*2 fut:τεμ*c aorp:τμηθ/ἐτμηθ perfm:τετμη",
    "τίκτω": "aor2:τεκ/ἐτεκ*2 fut:τεξ aorp:τεχθ/ἐτεχθ",
    "τρέπω": "aorp2:τραπ/ἐτραπ*2 aorp:τρεφθ/ἐτρεφθ",
    "τρέφω": "fut:θρεψ aor1:θρεψ/ἐθρεψ aorp2:τραφ/ἐτραφ*2 perfm:τεθραπ",
    "τρέχω": "aor2:δραμ/ἐδραμ*2 fut:δραμ*c",
    "τυγχάνω": "aor2:τυχ/ἐτυχ*2 perf:τετυχηκ perf:τετευχ",
    "ὑπισχνέομαι": "aor2:ὑποσχ/ὑπεσχ*2",
    "φαίνω": "fut:φαν*c aor1:φαν/ἐφαν aor1:φην/ἐφην aorp2:φαν/ἐφαν*2 perf:πεφην*2",
    "φέρω": "aor1:ἐνεγκ/ἠνεγκ aor2:ἐνεγκ/ἠνεγκ*2 fut:οἰσ aorp:ἐνεχθ/ἠνεχθ perf:ἐνηνοχ",
    "φεύγω": "aor2:φυγ/ἐφυγ*2 fut:φευξ perf:πεφευγ*2",
    "φθάνω": "fut:φθασ aor1:φθασ/ἐφθασ",
    "φθείρω": "fut:φθερ*c aor1:φθειρ/ἐφθειρ aorp2:φθαρ/ἐφθαρ*2 perfm:ἐφθαρ",
    "φύω": "fut:φυσ aor1:φυσ/ἐφυσ aorp2:φυ/ἐφυ*2 perf:πεφυκ",
    "χαίρω": "aorp2:χαρ/ἐχαρ*2 fut:χαρησ",
    "χέω": "fut:χε*c fut:χευσ aor1:χε/ἐχε aorp:χυθ/ἐχυθ perfm:κεχυ pres:χυνν",
    "ἀγγέλλω": "fut:ἀγγελ*c aor1:ἀγγειλ/ἠγγειλ aorp2:ἀγγελ/ἠγγελ*2 perfm:ἠγγελ perf:ἠγγελκ",
    "ἀμύνω": "fut:ἀμυν*c aor1:ἀμυν/ἠμυν",
    "ἅπτω": "fut:ἁψ aor1:ἁψ/ἡψ aorp:ἁφθ/ἡφθ",
    "ἀρνέομαι": "fut:ἀρνησ aor1:ἀρνησ/ἠρνησ aorp:ἀρνηθ/ἠρνηθ perfm:ἠρνη",
    "ἁρπάζω": "fut:ἁρπασ fut:ἁρπαξ aor1:ἁρπασ/ἡρπασ aor1:ἁρπαξ/ἡρπαξ aorp2:ἁρπαγ/ἡρπαγ*2 aorp:ἁρπασθ/ἡρπασθ",
    "ἀφανίζω": "fut:ἀφανι*c",
    "ἥκω": "fut:ἡξ aor1:ἡξ/ἡξ perf:ἡκ",
    "θραύω": "fut:θραυσ aor1:θραυσ/ἐθραυσ aorp:θραυσθ/ἐθραυσθ perfm:τεθραυσ",
    "κομίζω": "fut:κομι*c",
    "σπάω": "fut:σπασ aor1:σπασ/ἐσπασ aorp:σπασθ/ἐσπασθ perfm:ἐσπασ",
    "σφάζω": "fut:σφαξ aor1:σφαξ/ἐσφαξ aorp2:σφαγ/ἐσφαγ*2 perfm:ἐσφαγ",
    "τίνω": "fut:τεισ aor1:τεισ/ἐτεισ",
    "χράομαι": "fut:χρησ aor1:χρησ/ἐχρησ perfm:κεχρη",
    "ψάλλω": "fut:ψαλ*c aor1:ψαλ/ἐψαλ",
    "ἀλλάσσω": "fut:ἀλλαξ aor1:ἀλλαξ/ἠλλαξ aorp2:ἀλλαγ/ἠλλαγ*2",
    "πλάσσω": "fut:πλασ aor1:πλασ/ἐπλασ aorp:πλασθ/ἐπλασθ perfm:πεπλασ",
    "ἁρμόζω": "fut:ἁρμοσ aor1:ἁρμοσ/ἡρμοσ aorp:ἁρμοσθ/ἡρμοσθ perfm:ἡρμοσ",
    "παίζω": "fut:παιξ aor1:παιξ/ἐπαιξ",
    "στενάζω": "fut:στεναξ aor1:στεναξ/ἐστεναξ",
    "ἀλαλάζω": "fut:ἀλαλαξ aor1:ἀλαλαξ/ἠλαλαξ",
    "οἰμώζω": "fut:οἰμωξ aor1:οἰμωξ/ᾠμωξ",
    "κτίζω": "perf:ἐκτικ perfm:ἐκτισ",
    "δίδωμι": "fut:δωσ aorp:δοθ/ἐδοθ perf:δεδωκ perfm:δεδο",
    "τίθημι": "fut:θησ aorp:τεθ/ἐτεθ perf:τεθεικ perfm:τεθει fut:θη*w",
    "ἵστημι": "fut:στησ aor1:στησ/ἐστησ aorp:σταθ/ἐσταθ perfm:ἑστα pres:ἱσταν*w",
    "ἵημι": "fut:ἡσ aorp:ἑθ/ἑθ,εἱθ perf:εἱκ perfm:ἑω perfm:εἱ pres:ἱ*w pres:ἱ*wc",
    "πίμπλημι": "fut:πλησ aor1:πλησ/ἐπλησ aorp:πλησθ/ἐπλησθ perfm:πεπλησ",
    "ἐμπίμπλημι": "fut:πλησ aor1:πλησ/ἐπλησ aorp:πλησθ/ἐπλησθ perfm:πεπλησ",
    "φημί": "fut:φησ aor1:φησ/ἐφησ",
}

# ---------------------------------------------------------------------------
# Written-out paradigms

# Participle types: (NSM ending, masculine stem ending, feminine stem
# ending, NSN ending, dative plural); the accent stays on the ending's first
# syllable (διδούς -όντος -οῦσα, στάς -άντος -ᾶσα, ἑστώς -ῶτος -ῶσα).
_PARTICIPLE_TYPES = {
    "ους": ("ουσ", "οντ", "ουσ", "ον", "ουσι"),     # διδούς, δούς, γνούς
    "εις": ("εισ", "εντ", "εισ", "εν", "εισι"),     # τιθείς, θείς, εἵς
    "ας": ("ᾱσ", "αντ", "ᾱσ", "αν", "ᾱσι"),         # ἱστάς, στάς, βάς
    "υς": ("ῡσ", "υντ", "ῡσ", "υν", "ῡσι"),         # δεικνύς, δύς
    "ων": ("ων", "οντ", "ουσ", "ον", "ουσι"),       # ὤν, ἰών
    "ως": ("ωσ", "ωτ", "ωσ", "οσ", "ωσι"),          # ἑστώς, -ῶτος, -ῶσα, ἑστός
    "κως": ("ωσ", "οτ", "υι", "οσ", "οσι"),         # εἰδώς, -ότος, -υῖα
}


def _participle(body: str, kind: str, breathing: str | None = None) -> list[tuple[str, str]]:
    """(case-number-gender, accented form) of an athematic participle."""
    syllable = ph.nucleus_count(body)
    out = []
    for cng, ending, ultima, _ in _ptc_third(*_PARTICIPLE_TYPES[kind]):
        word = body + ending
        if breathing and not body:
            word = ph.set_breathing(word, breathing)
        out.append((cng, ph.finish(ph.accentuate(word, "syllable", ultima, 0, syllable, ph.ACUTE)[0])))
    return out


def _men_participle(stem_with_men: str, accent_syllable: int) -> list[tuple[str, str]]:
    """(cng, accented form) for a -μενος participle, accent persistent on a nucleus."""
    out = []
    for cng, ending, ultima, _ in _second_declension(""):
        word = stem_with_men + ending
        out.append((cng, ph.finish(ph.accentuate(word, "syllable", ultima, 0, accent_syllable)[0])))
    return out


TABLES: dict[str, str] = {}

TABLES["δίδωμι"] = """
PaI-1S δίδωμι; PaI-2S δίδωσ; PaI-3S δίδωσι δίδωσιν; PaI-1P δίδομεν; PaI-2P δίδοτε
PaI-3P διδόασι διδόασιν διδοῦσι! διδοῦσιν!
IaI-1S ἐδίδουν ἐδίδων; IaI-2S ἐδίδουσ; IaI-3S ἐδίδου; IaI-1P ἐδίδομεν; IaI-2P ἐδίδοτε; IaI-3P ἐδίδοσαν ἐδίδουν
PaS-1S διδῶ!; PaS-2S διδῷσ!; PaS-3S διδῷ!; PaS-1P διδῶμεν!; PaS-2P διδῶτε!; PaS-3P διδῶσι! διδῶσιν!
PaO-3S διδοίη!; PaM-2S δίδου; PaM-3S διδότω; PaM-2P δίδοτε; PaM-3P διδότωσαν
PaN διδόναι!; PaP @ους διδ
PmI-1S δίδομαι; PmI-2S δίδοσαι; PmI-3S δίδοται; PmI-1P διδόμεθα; PmI-2P δίδοσθε; PmI-3P δίδονται
ImI-1S ἐδιδόμην; ImI-3S ἐδίδοτο; ImI-3P ἐδίδοντο; PmS-3S διδῶται!; PmN δίδοσθαι!; PmP @μενος διδόμεν
AaI-1S ἔδωκα; AaI-2S ἔδωκασ; AaI-3S ἔδωκε ἔδωκεν; AaI-1P ἐδώκαμεν; AaI-2P ἐδώκατε; AaI-3P ἔδωκαν
2AaI-1P ἔδομεν; 2AaI-2P ἔδοτε; 2AaI-3P ἔδοσαν
2AaS-1S δῶ!; 2AaS-2S δῷσ!; 2AaS-3S δῷ! δοῖ! δώῃ!; 2AaS-1P δῶμεν!; 2AaS-2P δῶτε!; 2AaS-3P δῶσι! δῶσιν!
2AaO-1S δοίην!; 2AaO-2S δοίησ!; 2AaO-3S δοίη! δῴη!; 2AaO-3P δοῖεν!
2AaM-2S δόσ; 2AaM-3S δότω; 2AaM-2P δότε; 2AaM-3P δότωσαν
2AaN δοῦναι!; 2AaP @ους δ
2AsI-1S ἐδόμην; 2AsI-2S ἔδου; 2AsI-3S ἔδοτο; 2AsI-1P ἐδόμεθα; 2AsI-2P ἔδοσθε; 2AsI-3P ἔδοντο; 2AsS-1S δῶμαι!; 2AsS-3S δῶται!
2AsM-2S δοῦ; 2AsN δόσθαι!; 2AsP @μενος δόμεν
"""
TABLES["τίθημι"] = """
PaI-1S τίθημι; PaI-2S τίθησ; PaI-3S τίθησι τίθησιν; PaI-1P τίθεμεν; PaI-2P τίθετε; PaI-3P τιθέασι τιθέασιν
IaI-1S ἐτίθην; IaI-2S ἐτίθεισ; IaI-3S ἐτίθει; IaI-1P ἐτίθεμεν; IaI-2P ἐτίθετε; IaI-3P ἐτίθεσαν ἐτίθουν
PaS-1S τιθῶ!; PaS-3S τιθῇ!; PaS-3P τιθῶσι! τιθῶσιν!; PaM-2S τίθει; PaM-3S τιθέτω; PaM-2P τίθετε
PaN τιθέναι!; PaP @εις τιθ
PmI-1S τίθεμαι; PmI-2S τίθεσαι; PmI-3S τίθεται; PmI-1P τιθέμεθα; PmI-2P τίθεσθε; PmI-3P τίθενται
ImI-1S ἐτιθέμην; ImI-3S ἐτίθετο; ImI-3P ἐτίθεντο; PmN τίθεσθαι!; PmP @μενος τιθέμεν
AaI-1S ἔθηκα; AaI-2S ἔθηκασ; AaI-3S ἔθηκε ἔθηκεν; AaI-1P ἐθήκαμεν; AaI-2P ἐθήκατε; AaI-3P ἔθηκαν
2AaI-1P ἔθεμεν; 2AaI-2P ἔθετε; 2AaI-3P ἔθεσαν
2AaS-1S θῶ!; 2AaS-2S θῇσ!; 2AaS-3S θῇ!; 2AaS-1P θῶμεν!; 2AaS-2P θῆτε!; 2AaS-3P θῶσι! θῶσιν!
2AaO-3S θείη!; 2AaM-2S θέσ; 2AaM-3S θέτω; 2AaM-2P θέτε; 2AaM-3P θέτωσαν
2AaN θεῖναι!; 2AaP @εις θ
2AsI-1S ἐθέμην; 2AsI-2S ἔθου; 2AsI-3S ἔθετο; 2AsI-1P ἐθέμεθα; 2AsI-2P ἔθεσθε; 2AsI-3P ἔθεντο
2AsS-1S θῶμαι!; 2AsS-2S θῇ!; 2AsS-3S θῆται!; 2AsS-1P θώμεθα!; 2AsS-2P θῆσθε!; 2AsS-3P θῶνται!
2AsO-3S θεῖτο!; 2AsM-2S θοῦ; 2AsM-3S θέσθω; 2AsM-2P θέσθε; 2AsN θέσθαι!; 2AsP @μενος θέμεν
"""
TABLES["ἵημι"] = """
PaI-1S ἵημι; PaI-2S ἵησ; PaI-3S ἵησι ἵησιν; PaI-1P ἵεμεν; PaI-2P ἵετε; PaI-3P ἱᾶσι! ἱᾶσιν!
IaI-3S ἵει; PaM-2S ἵει; PaM-2P ἵετε; PaN ἱέναι!; PaP @εις ἱ
PmI-3S ἵεται; PmI-3P ἵενται; PmP @μενος ἱέμεν
AaI-1S ἧκα; AaI-2S ἧκασ ἧκεσ; AaI-3S ἧκε ἧκεν; AaI-1P ἥκαμεν; AaI-2P ἥκατε; AaI-3P ἧκαν
2AaS-1S ὧ!; 2AaS-2S ᾗσ!; 2AaS-3S ᾗ!; 2AaS-1P ὧμεν!; 2AaS-2P ἧτε!; 2AaS-3P ὧσι! ὧσιν!
2AaO-3S εἵη!; 2AaM-2S ἕσ; 2AaM-3S ἕτω; 2AaM-2P ἕτε; 2AaN εἷναι!; 2AaP @εις ~
"""
TABLES["ἵστημι"] = """
PaI-1S ἵστημι; PaI-2S ἵστησ; PaI-3S ἵστησι ἵστησιν; PaI-1P ἵσταμεν; PaI-2P ἵστατε; PaI-3P ἱστᾶσι! ἱστᾶσιν!
IaI-1S ἵστην; IaI-3S ἵστη; IaI-3P ἵστασαν; PaS-3S ἱστῇ!; PaM-2S ἵστη; PaM-2P ἵστατε
PaN ἱστάναι!; PaP @ας ἱστ
PmI-1S ἵσταμαι; PmI-2S ἵστασαι; PmI-3S ἵσταται; PmI-1P ἱστάμεθα; PmI-2P ἵστασθε; PmI-3P ἵστανται
ImI-1S ἱστάμην; ImI-3S ἵστατο; ImI-3P ἵσταντο; PmM-2S ἵστασο; PmM-2P ἵστασθε; PmN ἵστασθαι!
PmP @μενος ἱστάμεν
2AaI-1S ἔστην; 2AaI-2S ἔστησ; 2AaI-3S ἔστη; 2AaI-1P ἔστημεν; 2AaI-2P ἔστητε; 2AaI-3P ἔστησαν
2AaS-1S στῶ!; 2AaS-2S στῇσ!; 2AaS-3S στῇ!; 2AaS-1P στῶμεν!; 2AaS-2P στῆτε!; 2AaS-3P στῶσι! στῶσιν!
2AaO-3S σταίη!; 2AaM-2S στῆθι στῆ στᾶ; 2AaM-3S στήτω; 2AaM-2P στῆτε; 2AaM-3P στήτωσαν
2AaN στῆναι!; 2AaP @ας στ
RaI-1S ἕστηκα; RaI-2S ἕστηκασ; RaI-3S ἕστηκε ἕστηκεν; RaI-1P ἑστήκαμεν; RaI-2P ἑστήκατε
RaI-3P ἑστήκασι ἑστήκασιν ἑστᾶσι!; RaN ἑστάναι! ἑστηκέναι!; RaP @κως ἑστηκ; RaP @ως ἑστ
LaI-1S εἱστήκειν ἱστήκειν; LaI-3S εἱστήκει ἱστήκει; LaI-3P εἱστήκεισαν ἱστήκεισαν
RmI-3S ἕσταται; RmP @μενος ἑστάμεν
"""
# The verb "to be" and its compounds (πάρειμι, ἄπειμι, σύνειμι, ἔνειμι).
TABLES["εἰμί"] = """
PaI-1S εἰμί; PaI-2S εἶ; PaI-3S ἐστί ἐστίν ἔστι ἔστιν; PaI-1P ἐσμέν; PaI-2P ἐστέ; PaI-3P εἰσί εἰσίν
IaI-1S ἤμην ἦν; IaI-2S ἦσ ἦσθα; IaI-3S ἦν; IaI-1P ἦμεν ἤμεθα; IaI-2P ἦτε; IaI-3P ἦσαν
PaS-1S ὦ!; PaS-2S ᾖσ!; PaS-3S ᾖ!; PaS-1P ὦμεν!; PaS-2P ἦτε!; PaS-3P ὦσι! ὦσιν!
PaO-1S εἴην!; PaO-2S εἴησ!; PaO-3S εἴη!; PaO-1P εἴημεν! εἶμεν!; PaO-3P εἴησαν! εἶεν!
PaM-2S ἴσθι; PaM-3S ἔστω ἤτω; PaM-2P ἔστε; PaM-3P ἔστωσαν
PaN εἶναι!; PaP @ων ~
FfI-1S ἔσομαι; FfI-2S ἔσῃ; FfI-3S ἔσται!; FfI-1P ἐσόμεθα; FfI-2P ἔσεσθε; FfI-3P ἔσονται
FfN ἔσεσθαι!; FfP @μενος ἐσόμεν
"""
# εἶμι "go" (ἄπειμι, εἴσειμι, ἔξειμι, πρόσειμι, σύνειμι).
TABLES["εἶμι"] = """
PaI-1S εἶμι; PaI-2S εἶ; PaI-3S εἶσι εἶσιν; PaI-1P ἴμεν; PaI-2P ἴτε; PaI-3P ἴασι ἴασιν
IaI-1S ᾔειν; IaI-3S ᾔει ᾔειν; IaI-3P ᾖσαν ᾔεσαν
PaS-3S ἴῃ!; PaS-3P ἴωσι!; PaM-2S ἴθι; PaM-3S ἴτω; PaM-2P ἴτε; PaN ἰέναι!; PaP @ων ἰ
"""
TABLES["φημί"] = """
PaI-1S φημί; PaI-2S φήσ φῄσ; PaI-3S φησί φησίν; PaI-1P φαμέν; PaI-2P φατέ; PaI-3P φασί φασίν
IaI-1S ἔφην; IaI-2S ἔφησ ἔφησθα; IaI-3S ἔφη; IaI-1P ἔφαμεν; IaI-3P ἔφασαν
PaN φάναι!; PaP @ας φ
"""
TABLES["κεῖμαι"] = """
PmI-1S κεῖμαι; PmI-2S κεῖσαι; PmI-3S κεῖται; PmI-1P κείμεθα; PmI-2P κεῖσθε; PmI-3P κεῖνται
ImI-1S ἐκείμην; ImI-3S ἔκειτο; ImI-3P ἔκειντο; PmM-2S κεῖσο; PmM-3S κείσθω; PmN κεῖσθαι!
PmP @μενος κείμεν
FfI-3S κείσεται; FfI-3P κείσονται
"""
TABLES["κάθημαι"] = """
PmI-1S κάθημαι; PmI-2S κάθῃ κάθησαι; PmI-3S κάθηται; PmI-1P καθήμεθα; PmI-2P κάθησθε; PmI-3P κάθηνται
ImI-1S ἐκαθήμην; ImI-3S ἐκάθητο; ImI-3P ἐκάθηντο; PmM-2S κάθου; PmM-3S καθήσθω; PmN καθῆσθαι!
PmP @μενος καθήμεν; PmS-3S κάθηται!
"""
TABLES["δύναμαι"] = """
PmI-1S δύναμαι; PmI-2S δύνασαι δύνῃ; PmI-3S δύναται; PmI-1P δυνάμεθα; PmI-2P δύνασθε; PmI-3P δύνανται
ImI-1S ἐδυνάμην ἠδυνάμην; ImI-2S ἐδύνω; ImI-3S ἐδύνατο ἠδύνατο; ImI-1P ἐδυνάμεθα; ImI-3P ἐδύναντο ἠδύναντο
PmS-1S δύνωμαι!; PmS-3S δύνηται!; PmS-3P δύνωνται!; PmO-3S δύναιτο!; PmN δύνασθαι!
PmP @μενος δυνάμεν
"""
TABLES["ἐπίσταμαι"] = """
PmI-1S ἐπίσταμαι; PmI-2S ἐπίστασαι; PmI-3S ἐπίσταται; PmI-1P ἐπιστάμεθα; PmI-2P ἐπίστασθε; PmI-3P ἐπίστανται
ImI-3S ἠπίστατο; PmN ἐπίστασθαι!; PmP @μενος ἐπιστάμεν
"""
TABLES["οἶδα"] = """
2RaI-1S οἶδα; 2RaI-2S οἶδασ οἶσθα; 2RaI-3S οἶδε οἶδεν; 2RaI-1P οἴδαμεν ἴσμεν; 2RaI-2P οἴδατε ἴστε
2RaI-3P οἴδασι οἴδασιν ἴσασι ἴσασιν
2RaS-1S εἰδῶ!; 2RaS-3S εἰδῇ!; 2RaS-1P εἰδῶμεν!; 2RaS-2P εἰδῆτε!; 2RaS-3P εἰδῶσι! εἰδῶσιν!
2RaM-2S ἴσθι; 2RaM-2P ἴστε; 2RaN εἰδέναι!; 2RaP @κως εἰδ
2LaI-1S ᾔδειν ᾔδειμεν; 2LaI-2S ᾔδεισ; 2LaI-3S ᾔδει; 2LaI-2P ᾔδειτε; 2LaI-3P ᾔδεισαν
FaI-1S εἰδήσω; FaI-3S εἰδήσει; FaI-3P εἰδήσουσι εἰδήσουσιν; FfI-3S εἰδήσεται
"""
# Root (athematic) aorists.
TABLES["βαίνω"] = """
2AaI-1S ἔβην; 2AaI-2S ἔβησ; 2AaI-3S ἔβη; 2AaI-1P ἔβημεν; 2AaI-2P ἔβητε; 2AaI-3P ἔβησαν
2AaS-1S βῶ!; 2AaS-2S βῇσ!; 2AaS-3S βῇ!; 2AaS-1P βῶμεν!; 2AaS-2P βῆτε!; 2AaS-3P βῶσι! βῶσιν!
2AaO-3S βαίη!; 2AaM-2S βῆθι βᾶ; 2AaM-3S βήτω; 2AaM-2P βῆτε; 2AaM-3P βήτωσαν
2AaN βῆναι!; 2AaP @ας β
"""
TABLES["γινώσκω"] = """
2AaI-1S ἔγνων; 2AaI-2S ἔγνωσ; 2AaI-3S ἔγνω; 2AaI-1P ἔγνωμεν; 2AaI-2P ἔγνωτε; 2AaI-3P ἔγνωσαν
2AaS-1S γνῶ!; 2AaS-2S γνῷσ!; 2AaS-3S γνῷ! γνοῖ!; 2AaS-1P γνῶμεν!; 2AaS-2P γνῶτε!; 2AaS-3P γνῶσι! γνῶσιν!
2AaO-3S γνοίη!; 2AaM-2S γνῶθι; 2AaM-3S γνώτω; 2AaM-2P γνῶτε; 2AaM-3P γνώτωσαν
2AaN γνῶναι!; 2AaP @ους γν
"""
TABLES["γιγνώσκω"] = TABLES["γινώσκω"]
TABLES["εἴδω"] = TABLES["οἶδα"]
TABLES["δύω"] = """
2AaI-1S ἔδυν; 2AaI-3S ἔδυ; 2AaI-3P ἔδυσαν; 2AaS-3S δύῃ; 2AaN δῦναι!; 2AaP @υς δ
"""
TABLES["δύνω"] = TABLES["δύω"]
# Impersonal verbs: third persons only.
TABLES["δεῖ"] = """
PaI-3S δεῖ!; PaS-3S δέῃ!; PaO-3S δέοι!; PaN δεῖν!; IaI-3S ἔδει!; FaI-3S δεήσει; AaI-3S ἐδέησε ἐδέησεν
PaP-NSN δέον!; PaP-ASN δέον!; PaP-NPN δέοντα!; PaP-APN δέοντα!; PaP-GSN δέοντος!
"""
TABLES["ἔξεστι"] = """
PaI-3S ἔξεστι ἔξεστιν; IaI-3S ἐξῆν!; PaP-NSN ἐξόν!; PaP-ASN ἐξόν!; PaN ἐξεῖναι!; PaS-3S ἐξῇ!; FfI-3S ἐξέσται!
"""
TABLES["ἔξεστιν"] = TABLES["ἔξεστι"]
TABLES["πρέπει"] = """
PaI-3S πρέπει; IaI-3S ἔπρεπε ἔπρεπεν; PaP-NSN πρέπον!; PaP-ASN πρέπον!; PaN πρέπειν!; PaS-3S πρέπῃ!
"""
# ἔχω's second aorist imperative σχές (πρόσχες, παράσχες, ἐπίσχες), like θές, δός.
TABLES["ἔχω"] = """
2AaM-2S σχέσ; 2AaM-2P σχέτε; 2AaM-3S σχέτω
"""
TABLES["διδράσκω"] = """
2AaI-1S ἔδραν; 2AaI-2S ἔδρασ; 2AaI-3S ἔδρα; 2AaI-1P ἔδραμεν; 2AaI-3P ἔδρασαν; 2AaS-3S δρᾷ!; 2AaN δρᾶναι!
2AaP @ας δρ
"""
TABLES["φύω"] = """
2AaI-3S ἔφυ; 2AaI-3P ἔφυσαν; 2AaN φῦναι!; 2AaP @υς φ
"""

_CODE = re.compile(r"^(?P<tense>2?[PIFARL])(?P<slot>[amfspq])(?P<mood>[ISOMNP])(?:-(?P<rest>[0-9A-Z]+))?$")


def table_forms(lemma: str) -> tuple:
    """A written-out paradigm as `Stem.forms` entries (see `morph_verbs`)."""
    text = TABLES.get(lemma)
    if not text:
        return ()
    entries = []
    slot_names = {"a": "act", "m": "mp", "f": "fm", "s": "am", "p": "ap", "q": "fp"}
    for chunk in re.split(r"[;\n]", text):
        fields = chunk.split()
        if not fields:
            continue
        match = _CODE.match(fields[0])
        if not match:
            raise ValueError(f"bad code {fields[0]!r} in the table of {lemma}")
        tense, slot, mood, rest = match["tense"], slot_names[match["slot"]], match["mood"], match["rest"] or ""
        if fields[1] == "@μενος":
            body = fields[2]
            plain = ph.plain(body)
            syllable = ph.accent_position(body)[0]
            for cng, form in _men_participle(plain, syllable):
                entries.append((tense, slot, "P", cng, form, True, 1.0))
            continue
        if fields[1].startswith("@"):
            kind = fields[1][1:]
            body = fields[2] if len(fields) > 2 else "~"
            breathing = None
            if body.startswith("~"):
                body = body[1:]
                breathing = ph.ROUGH if ph.initial_breathing(lemma) == ph.ROUGH and not body else ph.SMOOTH
                if body:
                    breathing = None
            for cng, form in _participle(ph.plain(body), kind, breathing):
                entries.append((tense, slot, "P", cng, form, True, 1.0))
            continue
        for form in fields[1:]:
            fixed = form.endswith("!")
            form = form.rstrip("!")
            from lxx_tagging.text import normalize
            entries.append((tense, slot, mood, rest, normalize(form), fixed, 1.0))
    return tuple(entries)


def table_stem(lemma: str) -> Stem | None:
    forms = table_forms(lemma)
    if not forms:
        return None
    return Stem("table", ph.plain(lemma), (), 0.95, "table", forms=forms)


_PART = re.compile(r"^(?P<system>[a-z0-9]+):(?P<stem>[^/*]+)(?:/(?P<aug>[^*]+))?(?:\*(?P<flags>\w+))?$")


def principal_parts(lemma: str, weight: float = 0.9) -> list[Stem]:
    """The PRINCIPAL_PARTS entry of a simplex as Stems."""
    spec = PRINCIPAL_PARTS.get(lemma)
    if not spec:
        return []
    stems = []
    for part in spec.split():
        match = _PART.match(part)
        if not match:
            raise ValueError(f"bad principal part {part!r} for {lemma}")
        flags = match["flags"] or ""
        system = match["system"]
        stem = ph.plain(match["stem"])
        augmented = tuple((ph.plain(a), 1.0) for a in match["aug"].split(",")) if match["aug"] else ()
        part_weight = weight * (0.5 if "w" in flags else 1.0)
        contract = "ε" if "c" in flags else None
        if contract:
            stem = stem + "ε"        # βαλ-ε-: βαλῶ, βαλεῖς
        if system == "impf":
            stems.append(Stem("impf", stem, tuple((ph.plain(a), 1.0) for a in match["stem"].split(",")),
                              part_weight, "irregular"))
            continue
        if system == "pres" and not contract:
            contract = None
        elif system == "pres":
            # A Koine contract present beside the athematic one: συνίημι's
            # συνιῶ, συνιεῖ, συνιοῦσι (pres:ἱ*wc).
            pass
        stems.append(Stem(system, stem, augmented, part_weight, "irregular", contract, "2" in flags))
    return stems


def nymi_forms(stem: str) -> tuple:
    """The athematic present system of a -νυμι verb (δείκνυμι, σβέννυμι,
    ὄμνυμι, στρώννυμι) from its present stem in -νυ- (plain), as
    `Stem.forms` entries: long ῡ in the singular active, short elsewhere."""
    head = stem[:-1]                     # δεικν
    long_stem, short_stem = head + "ῡ", head + "υ"
    rows = [
        ("P", "act", "I", "1S", long_stem + "μι"), ("P", "act", "I", "2S", long_stem + "σ"),
        ("P", "act", "I", "3S", long_stem + "σι"), ("P", "act", "I", "3S", long_stem + "σιν"),
        ("P", "act", "I", "1P", short_stem + "μεν"), ("P", "act", "I", "2P", short_stem + "τε"),
        ("P", "act", "I", "3P", short_stem + "ασι"), ("P", "act", "I", "3P", short_stem + "ασιν"),
        ("P", "act", "M", "2S", long_stem), ("P", "act", "M", "3S", short_stem + "τω"),
        ("P", "act", "M", "2P", short_stem + "τε"),
        ("P", "mp", "I", "1S", short_stem + "μαι"), ("P", "mp", "I", "2S", short_stem + "σαι"),
        ("P", "mp", "I", "3S", short_stem + "ται"), ("P", "mp", "I", "1P", short_stem + "μεθα"),
        ("P", "mp", "I", "2P", short_stem + "σθε"), ("P", "mp", "I", "3P", short_stem + "νται"),
        ("P", "mp", "M", "2S", short_stem + "σο"), ("P", "mp", "M", "2P", short_stem + "σθε"),
    ]
    imperfect = [
        ("I", "act", "I", "1S", long_stem + "ν"), ("I", "act", "I", "2S", long_stem + "σ"),
        ("I", "act", "I", "3S", long_stem), ("I", "act", "I", "1P", short_stem + "μεν"),
        ("I", "act", "I", "2P", short_stem + "τε"), ("I", "act", "I", "3P", short_stem + "σαν"),
        ("I", "mp", "I", "1S", short_stem + "μην"), ("I", "mp", "I", "3S", short_stem + "το"),
        ("I", "mp", "I", "3P", short_stem + "ντο"),
    ]
    entries = []
    for tense, slot, mood, rest, form in rows:
        entries.append((tense, slot, mood, rest, ph.finish(ph.accentuate(form, "recessive")[0]), False, 0.9))
    for tense, slot, mood, rest, form in imperfect:
        for augmented, weight in ph.augment(form):
            entries.append((tense, slot, mood, rest, ph.finish(ph.accentuate(augmented, "recessive")[0]), False,
                            0.9 * weight))
    nuclei = ph.nucleus_count(short_stem)
    entries.append(("P", "act", "N", "", ph.finish(ph.accentuate(short_stem + "ναι", "syllable", None, 0,
                                                                  nuclei - 1)[0]), True, 0.9))
    entries.append(("P", "mp", "N", "", ph.finish(ph.accentuate(short_stem + "σθαι", "recessive")[0]), True, 0.9))
    for cng, form in _participle(head, "υς"):
        entries.append(("P", "act", "P", cng, form, True, 0.9))
    for cng, form in _men_participle(short_stem + "μεν", nuclei - 1):
        entries.append(("P", "mp", "P", cng, form, True, 0.9))
    return tuple(entries)


# ---------------------------------------------------------------------------
# Irregular nouns and adjectives, written out: lemma -> (gender or "A" for
# an adjective, forms). A noun's slots are case-number (GS:ἀνδρός), an
# adjective's case-number-gender (GSM:παντός). Compounds of πᾶς and εἷς
# (ἅπας, σύμπας; οὐδείς, μηδείς) are formed from these by `prefixed`.
IRREGULAR_NOMINALS: dict[str, tuple[str, str]] = {
    # Added for Septuagint forms the rules missed (round 2): Attic γήρως,
    # the contracted χειμάρρους beside χείμαρρος, τετράπους's dative plural
    # τετράποσι, δεκατέσσαρες's other cases.
    "γῆρας": ("N", "NS:γῆρας AS:γῆρας GS:γήρως GS:γήρους DS:γήρει DS:γήρᾳ"),
    "χείμαρρος": ("M", "NS:χείμαρρος NS:χειμάρρους GS:χειμάρρου DS:χειμάρρῳ AS:χείμαρρον AS:χειμάρρουν "
                       "NP:χείμαρροι GP:χειμάρρων DP:χειμάρροις AP:χειμάρρους"),
    "τετράπους": ("A", "NSM:τετράπους NSF:τετράπους NSN:τετράπουν ASN:τετράπουν GSM:τετράποδος GSF:τετράποδος "
                       "GSN:τετράποδος DSM:τετράποδι DSN:τετράποδι ASM:τετράποδα ASF:τετράποδα NPM:τετράποδες "
                       "NPF:τετράποδες NPN:τετράποδα APN:τετράποδα APM:τετράποδας APF:τετράποδας GPM:τετραπόδων "
                       "GPN:τετραπόδων GPF:τετραπόδων DPM:τετράποσι DPM:τετράποσιν DPN:τετράποσι DPN:τετράποσιν "
                       "DPF:τετράποσι DPF:τετράποσιν"),
    "ἀρήν": ("M", "NS:ἀρήν GS:ἀρνός DS:ἀρνί AS:ἄρνα NP:ἄρνες GP:ἀρνῶν DP:ἄρνασι DP:ἄρνασιν DP:ἀρνάσι AP:ἄρνας"),
    "θίς": ("M", "NS:θίς GS:θινός DS:θινί AS:θῖνα NP:θῖνες GP:θινῶν DP:θισί DP:θισίν AP:θῖνας"),
    # TBESG files πενία as neuter; it is a feminine of the first declension.
    "πενία": ("F", "NS:πενία GS:πενίας DS:πενίᾳ AS:πενίαν"),
    # TBESG files ὀκτακόσιοι as a masculine noun; it is the numeral adjective.
    "ὀκτακόσιοι": ("A", "NPM:ὀκτακόσιοι NPF:ὀκτακόσιαι NPN:ὀκτακόσια APM:ὀκτακοσίους APF:ὀκτακοσίας "
                        "APN:ὀκτακόσια GPM:ὀκτακοσίων GPF:ὀκτακοσίων GPN:ὀκτακοσίων DPM:ὀκτακοσίοις "
                        "DPF:ὀκτακοσίαις DPN:ὀκτακοσίοις"),
    "δεκατέσσαρες": ("A", "NPM:δεκατέσσαρες NPF:δεκατέσσαρες NPN:δεκατέσσαρα APM:δεκατέσσαρας APF:δεκατέσσαρας "
                          "APN:δεκατέσσαρα GPM:δεκατεσσάρων GPF:δεκατεσσάρων GPN:δεκατεσσάρων DPM:δεκατέσσαρσι "
                          "DPF:δεκατέσσαρσι DPN:δεκατέσσαρσι"),
    "ἀνήρ": ("M", "NS:ἀνήρ GS:ἀνδρός DS:ἀνδρί AS:ἄνδρα VS:ἄνερ NP:ἄνδρες VP:ἄνδρες GP:ἀνδρῶν DP:ἀνδράσι "
                  "DP:ἀνδράσιν AP:ἄνδρας"),
    "γυνή": ("F", "NS:γυνή GS:γυναικός DS:γυναικί AS:γυναῖκα VS:γύναι NP:γυναῖκες VP:γυναῖκες GP:γυναικῶν "
                  "DP:γυναιξί DP:γυναιξίν AP:γυναῖκας"),
    "πατήρ": ("M", "NS:πατήρ GS:πατρός DS:πατρί AS:πατέρα VS:πάτερ NP:πατέρες VP:πατέρες GP:πατέρων "
                   "DP:πατράσι DP:πατράσιν AP:πατέρας"),
    "μήτηρ": ("F", "NS:μήτηρ GS:μητρός DS:μητρί AS:μητέρα VS:μῆτερ NP:μητέρες GP:μητέρων DP:μητράσι "
                   "DP:μητράσιν AP:μητέρας"),
    "θυγάτηρ": ("F", "NS:θυγάτηρ GS:θυγατρός DS:θυγατρί AS:θυγατέρα VS:θύγατερ NP:θυγατέρες VP:θυγατέρες "
                     "GP:θυγατέρων DP:θυγατράσι DP:θυγατράσιν AP:θυγατέρας"),
    "γαστήρ": ("F", "NS:γαστήρ GS:γαστρός DS:γαστρί AS:γαστέρα NP:γαστέρες GP:γαστέρων DP:γαστράσι AP:γαστέρας"),
    "χείρ": ("F", "NS:χείρ GS:χειρός DS:χειρί AS:χεῖρα NP:χεῖρες GP:χειρῶν DP:χερσί DP:χερσίν DP:χειρσί "
                  "AP:χεῖρας"),
    "κύων": ("M", "NS:κύων GS:κυνός DS:κυνί AS:κύνα NP:κύνες GP:κυνῶν DP:κυσί DP:κυσίν AP:κύνας"),
    "οὖς": ("N", "NS:οὖς AS:οὖς GS:ὠτός DS:ὠτί NP:ὦτα AP:ὦτα GP:ὤτων DP:ὠσί DP:ὠσίν"),
    "ὕδωρ": ("N", "NS:ὕδωρ AS:ὕδωρ GS:ὕδατος DS:ὕδατι NP:ὕδατα AP:ὕδατα GP:ὑδάτων DP:ὕδασι DP:ὕδασιν"),
    "πῦρ": ("N", "NS:πῦρ AS:πῦρ GS:πυρός DS:πυρί NP:πυρά AP:πυρά GP:πυρῶν DP:πυροῖς"),
    "φῶς": ("N", "NS:φῶς AS:φῶς GS:φωτός DS:φωτί NP:φῶτα AP:φῶτα GP:φώτων DP:φωσί DP:φωσίν"),
    "θρίξ": ("F", "NS:θρίξ GS:τριχός DS:τριχί AS:τρίχα NP:τρίχες GP:τριχῶν DP:θριξί DP:θριξίν AP:τρίχας"),
    "ναῦς": ("F", "NS:ναῦς GS:νεώς GS:νηός DS:νηί AS:ναῦν NP:νῆες GP:νεῶν DP:ναυσί DP:ναυσίν AP:ναῦς"),
    "βοῦς": ("M", "NS:βοῦς GS:βοός DS:βοΐ AS:βοῦν NP:βόες GP:βοῶν DP:βουσί DP:βουσίν AP:βόας AP:βοῦς"),
    "γόνυ": ("N", "NS:γόνυ AS:γόνυ GS:γόνατος DS:γόνατι NP:γόνατα AP:γόνατα GP:γονάτων DP:γόνασι DP:γόνασιν"),
    "δόρυ": ("N", "NS:δόρυ AS:δόρυ GS:δόρατος DS:δόρατι NP:δόρατα AP:δόρατα GP:δοράτων DP:δόρασι"),
    "κέρας": ("N", "NS:κέρας AS:κέρας GS:κέρατος GS:κέρως DS:κέρατι DS:κέρᾳ NP:κέρατα NP:κέρα AP:κέρατα AP:κέρα "
                   "GP:κεράτων GP:κερῶν DP:κέρασι DP:κέρασιν"),
    "τέρας": ("N", "NS:τέρας AS:τέρας GS:τέρατος DS:τέρατι NP:τέρατα AP:τέρατα GP:τεράτων DP:τέρασι DP:τέρασιν"),
    "κρέας": ("N", "NS:κρέας AS:κρέας GS:κρέως GS:κρέατος DS:κρέατι NP:κρέα AP:κρέα GP:κρεῶν DP:κρέασι DP:κρέασιν"),
    "γάλα": ("N", "NS:γάλα AS:γάλα GS:γάλακτος DS:γάλακτι"),
    "μέλι": ("N", "NS:μέλι AS:μέλι GS:μέλιτος DS:μέλιτι"),
    "παῖς": ("M", "NS:παῖς GS:παιδός DS:παιδί AS:παῖδα VS:παῖ NP:παῖδες VP:παῖδες GP:παίδων DP:παισί DP:παισίν "
                  "AP:παῖδας"),
    "νοῦς": ("M", "NS:νοῦς GS:νοός GS:νοῦ DS:νοΐ DS:νῷ AS:νοῦν"),
    "πλοῦς": ("M", "NS:πλοῦς GS:πλοός GS:πλοῦ DS:πλοΐ AS:πλοῦν"),
    "χοῦς": ("M", "NS:χοῦς GS:χοός GS:χοῦ DS:χοΐ AS:χοῦν"),
    "ἀστήρ": ("M", "NS:ἀστήρ GS:ἀστέρος DS:ἀστέρι AS:ἀστέρα NP:ἀστέρες GP:ἀστέρων DP:ἀστράσι DP:ἀστράσιν "
                   "AP:ἀστέρας"),
    "μάρτυς": ("M", "NS:μάρτυς GS:μάρτυρος DS:μάρτυρι AS:μάρτυρα NP:μάρτυρες GP:μαρτύρων DP:μάρτυσι "
                    "DP:μάρτυσιν AP:μάρτυρας"),
    "γῆ": ("F", "NS:γῆ VS:γῆ GS:γῆς DS:γῇ AS:γῆν NP:γαῖ GP:γῶν AP:γᾶς"),
    "ἅλς": ("M", "NS:ἅλς GS:ἁλός DS:ἁλί AS:ἅλα NP:ἅλες GP:ἁλῶν"),
    "Ζεύς": ("M", "NS:Ζεύς GS:Διός DS:Διί AS:Δία"),
    "ὄρνις": ("F", "NS:ὄρνις GS:ὄρνιθος DS:ὄρνιθι AS:ὄρνιθα AS:ὄρνιν NP:ὄρνιθες GP:ὀρνίθων DP:ὄρνισι AP:ὄρνιθας"),
    "χάρις": ("F", "NS:χάρις GS:χάριτος DS:χάριτι AS:χάριν AS:χάριτα NP:χάριτες GP:χαρίτων DP:χάρισι AP:χάριτας"),
    "πᾶς": ("A", "NSM:πᾶς VSM:πᾶς GSM:παντός DSM:παντί ASM:πάντα NPM:πάντες VPM:πάντες GPM:πάντων DPM:πᾶσι "
                 "DPM:πᾶσιν APM:πάντας NSF:πᾶσα VSF:πᾶσα GSF:πάσης DSF:πάσῃ ASF:πᾶσαν NPF:πᾶσαι GPF:πασῶν "
                 "DPF:πάσαις APF:πάσας NSN:πᾶν ASN:πᾶν VSN:πᾶν GSN:παντός DSN:παντί NPN:πάντα APN:πάντα VPN:πάντα "
                 "GPN:πάντων DPN:πᾶσι DPN:πᾶσιν"),
    "εἷς": ("A", "NSM:εἷς GSM:ἑνός DSM:ἑνί ASM:ἕνα NSF:μία GSF:μιᾶς DSF:μιᾷ ASF:μίαν NSN:ἕν ASN:ἕν GSN:ἑνός "
                 "DSN:ἑνί"),
    # TAGNT codes δύο itself A-NUI, whatever its case.
    "δύο": ("A", "NUI:δύο DPM:δυσί DPM:δυσίν DPF:δυσί DPN:δυσί GPM:δυεῖν"),
    "τρεῖς": ("A", "NPM:τρεῖς APM:τρεῖς GPM:τριῶν DPM:τρισί DPM:τρισίν NPF:τρεῖς APF:τρεῖς GPF:τριῶν DPF:τρισί "
                   "DPF:τρισίν NPN:τρία APN:τρία GPN:τριῶν DPN:τρισί DPN:τρισίν"),
    "τέσσαρες": ("A", "NPM:τέσσαρες APM:τέσσαρας APM:τέσσαρες GPM:τεσσάρων DPM:τέσσαρσι DPM:τέσσαρσιν "
                      "DPM:τέσσαρσιν NPF:τέσσαρες APF:τέσσαρας GPF:τεσσάρων DPF:τέσσαρσι NPN:τέσσαρα APN:τέσσαρα "
                      "GPN:τεσσάρων DPN:τέσσαρσι"),
    "μέγας": ("A", "NSM:μέγας VSM:μέγας GSM:μεγάλου DSM:μεγάλῳ ASM:μέγαν NPM:μεγάλοι GPM:μεγάλων DPM:μεγάλοις "
                   "APM:μεγάλους NSF:μεγάλη GSF:μεγάλης DSF:μεγάλῃ ASF:μεγάλην NPF:μεγάλαι GPF:μεγάλων "
                   "DPF:μεγάλαις APF:μεγάλας NSN:μέγα ASN:μέγα GSN:μεγάλου DSN:μεγάλῳ NPN:μεγάλα APN:μεγάλα "
                   "GPN:μεγάλων DPN:μεγάλοις VSM:μεγάλε"),
    "πολύς": ("A", "NSM:πολύς GSM:πολλοῦ DSM:πολλῷ ASM:πολύν NPM:πολλοί GPM:πολλῶν DPM:πολλοῖς APM:πολλούς "
                   "NSF:πολλή GSF:πολλῆς DSF:πολλῇ ASF:πολλήν NPF:πολλαί GPF:πολλῶν DPF:πολλαῖς APF:πολλάς "
                   "NSN:πολύ ASN:πολύ GSN:πολλοῦ DSN:πολλῷ NPN:πολλά APN:πολλά GPN:πολλῶν DPN:πολλοῖς"),
    "ἵλεως": ("A", "NSM:ἵλεως NSF:ἵλεως NSN:ἵλεων ASM:ἵλεων ASF:ἵλεων ASN:ἵλεων GSM:ἵλεω DSM:ἵλεῳ NPM:ἵλεῳ "
                   "APM:ἵλεως"),
    "πρᾶος": ("A", "NSM:πρᾶος NSM:πραΰς GSM:πραέος GSM:πράου DSM:πραεῖ DSM:πράῳ ASM:πρᾶον ASM:πραΰν NPM:πραεῖς "
                   "NPM:πρᾶοι GPM:πραέων DPM:πραέσι APM:πραεῖς NSF:πραεῖα GSF:πραείας ASF:πραεῖαν NSN:πρᾶον "
                   "NSN:πραΰ APN:πραέα"),
}
# Comparison TAGNT files under the positive: positive -> (form, degree) nominatives.
IRREGULAR_COMPARISON: dict[str, tuple[tuple[str, str], ...]] = {
    "μέγας": (("μείζων", "C"), ("μειζότερος", "C"), ("μέγιστος", "S")),
    "πολύς": (("πλείων", "C"), ("πλέων", "C"), ("πλεῖστος", "S")),
    "ἀγαθός": (("κρείσσων", "C"), ("κρείττων", "C"), ("βελτίων", "C"), ("ἀμείνων", "C"), ("κράτιστος", "S"),
               ("βέλτιστος", "S"), ("ἄριστος", "S")),
    "κακός": (("χείρων", "C"), ("κακίων", "C"), ("χείριστος", "S"), ("κάκιστος", "S")),
    "μικρός": (("ἐλάσσων", "C"), ("ἐλάττων", "C"), ("μικρότερος", "C"), ("ἐλάχιστος", "S")),
    "ὀλίγος": (("ἐλάσσων", "C"), ("ὀλιγώτερος", "C"), ("ὀλίγιστος", "S")),
    "ταχύς": (("θάσσων", "C"), ("ταχύτερος", "C"), ("τάχιστος", "S")),
    "καλός": (("καλλίων", "C"), ("κάλλιστος", "S")),
    "ἡδύς": (("ἡδίων", "C"), ("ἥδιστος", "S")),
    "αἰσχρός": (("αἰσχίων", "C"), ("αἴσχιστος", "S")),
    "ἐχθρός": (("ἐχθίων", "C"), ("ἔχθιστος", "S")),
    "ῥᾴδιος": (("ῥᾴων", "C"), ("ῥᾷστος", "S")),
    "πρέσβυς": (("πρεσβύτερος", "C"), ("πρεσβύτατος", "S")),
    "ἐλαφρός": (("ἐλαφρότερος", "C"),),
}


def irregular_nominal(lemma_key: str) -> tuple[str, list[tuple[str, str]]] | None:
    """(gender or "A", [(slot, accented key)]) of an irregular noun or
    adjective, also for πᾶς's and εἷς's compounds (ἅπας, σύμπας, οὐδείς,
    μηδείς), whose accent follows the rules from the simplex's forms."""
    from lxx_tagging.text import normalize
    by_key = {normalize(name): name for name in IRREGULAR_NOMINALS}
    if lemma_key in by_key:
        gender, text = IRREGULAR_NOMINALS[by_key[lemma_key]]
        return gender, [(item.split(":")[0], normalize(item.split(":")[1])) for item in text.split()]
    letters = ph.letters(lemma_key)
    for prefix, simplex in (("ἁ", "πᾶς"), ("συμ", "πᾶς"), ("οὐδ", "εἷς"), ("μηδ", "εἷς"), ("οὐθ", "εἷς"),
                            ("μηθ", "εἷς")):
        simplex_letters = ph.letters(simplex)
        if letters == ph.letters(prefix) + simplex_letters:
            gender, text = IRREGULAR_NOMINALS[simplex]
            forms = []
            for item in text.split():
                slot, form = item.split(":")
                form = normalize(form)
                position = ph.accent_position(form)
                spelt = ph.plain(prefix)
                if simplex == "εἷς" and not ph.starts_with_vowel(form):
                    spelt += "ε"                       # οὐδε-μία
                body = spelt + ph.strip_breathing(ph.plain(form))
                if simplex == "εἷς":
                    # The simplex's accent stays on its syllable (οὐδενός, οὐδένα,
                    # οὐδεμία), acute where the simplex is a monosyllable (οὐδείς, οὐδέν).
                    kind = position[1] if ph.nucleus_count(form) > 1 else ph.ACUTE
                    accented = ph.accentuate(body, "syllable", None, 0,
                                             ph.nucleus_count(spelt) + position[0], kind)
                else:
                    accented = ph.accentuate(body, "syllable", "L" if slot in ("NSM", "VSM") else None, 0, 0)
                forms.append((slot, ph.finish(accented[0])))
            return gender, forms
    return None

#ifndef ALIRSNDAUGHTERSELECTOR_H
#define ALIRSNDAUGHTERSELECTOR_H

#include <TNamed.h>
#include <TClonesArray.h>
#include <TObjArray.h>

class TEntryList;
class TList;

class AliRsnCutSet;
class AliRsnEvent;
class AliRsnAction;

class AliRsnDaughterSelector : public TNamed {

public:

   AliRsnDaughterSelector(const char *name = "name", const char *title = "title");
   AliRsnDaughterSelector(const AliRsnDaughterSelector &copy);
   AliRsnDaughterSelector &operator=(const AliRsnDaughterSelector &copy);
   virtual ~AliRsnDaughterSelector();

   void          Init();
   void          InitActions(TList *list);
   void          Reset();
   Int_t         Add(AliRsnCutSet *cuts, Bool_t charged);
   Int_t         GetID(const char *cutSetName, Bool_t charged);
   TEntryList   *GetSelected(Int_t i, Char_t charge);
   TEntryList   *GetSelected(Int_t i, Short_t charge);
   void          ScanEvent(AliRsnEvent *ev);
   void          ExecActions(AliRsnEvent *ev);

   virtual void  Print(Option_t *option = "") const;

   TClonesArray *GetCutSetC() {return &fCutSetsC;}
   TClonesArray *GetCutSetN() {return &fCutSetsN;}

   void          AddAction(AliRsnAction *action);
   TObjArray    *GetActions() { return &fActions; }

   void SetLabelCheck(Bool_t useLabelCheck = kTRUE) { fUseLabelCheck = useLabelCheck;}

private:

   TClonesArray fCutSetsN;        // cuts for neutral daughters
   TClonesArray fCutSetsC;        // cuts for charged daughters (usually, the same)

   TClonesArray fEntryListsN;     // entry lists for neutrals
   TClonesArray fEntryListsP;     // entry lists for charged (one per sign)
   TClonesArray fEntryListsM;     // entry lists for charged (one per sign)

   Bool_t       fUseLabelCheck;   // flag is reapiting of label should be checked

   TObjArray    fActions;

   ClassDef(AliRsnDaughterSelector, 3)
};

#endif

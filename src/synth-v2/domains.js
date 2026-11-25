// Each tuple identifies the sampled object, exhaustive labels, three readings
// and a setting-specific aside. The same reasoning families span all domains.
export const DOMAINS = {
  medical: [
    ['screening record', 'Cedar', 'Birch', 'strip marker', 'assay flag', 'review alert', 'The clinic archive was reorganized last month.'],
    ['diagnostic chart', 'Aster', 'Iris', 'scan finding', 'culture marker', 'review flag', 'The notes were transcribed after rounds.'],
    ['laboratory specimen', 'Orchid', 'Fern', 'dye signal', 'reader flag', 'audit alert', 'The specimen log was printed on recycled paper.'],
  ],
  industrial: [
    ['turbine panel', 'Copper', 'Zinc', 'relay lamp', 'cabinet buzzer', 'console beacon', 'The maintenance binder had a blue cover.'],
    ['substation panel', 'Quarry', 'Slate', 'voltage flag', 'breaker lamp', 'dispatch alarm', 'The cabinet doors were repainted in spring.'],
    ['pumping station panel', 'Flint', 'Sand', 'pressure light', 'flow siren', 'monitor alarm', 'The walkways were resurfaced last year.'],
  ],
  logistics: [
    ['parcel', 'Elm', 'Pine', 'dock scan', 'sort flag', 'routing alert', 'The depot changed its stationery.'],
    ['pallet', 'Maple', 'Alder', 'seal flag', 'intake mark', 'dispatch alert', 'The manifest was printed on recycled paper.'],
    ['shipment', 'Spruce', 'Oak', 'weight alert', 'staging scan', 'release flag', 'The warehouse installed new shelving.'],
  ],
  finance: [
    ['ledger entry', 'Quartz', 'Opal', 'variance flag', 'reconciliation alert', 'control mark', 'The audit workbook was archived on Friday.'],
    ['invoice', 'Topaz', 'Ruby', 'duplicate marker', 'approval flag', 'payment alert', 'The report was formatted for landscape printing.'],
    ['transaction record', 'Amber', 'Jade', 'threshold flag', 'screening alert', 'review tag', 'The team stored the memos alphabetically.'],
  ],
  legal: [
    ['evidence file', 'Reed', 'Ash', 'custody flag', 'seal mark', 'review alert', 'The clerk filed the cover sheet separately.'],
    ['exhibit', 'Hazel', 'Holly', 'intake stamp', 'authentication mark', 'review flag', 'The archive issued replacement folders.'],
    ['case packet', 'Larch', 'Beech', 'filing indicator', 'verification stamp', 'docket alert', 'The hearing room was on the ground floor.'],
  ],
  abstract: [
    ['panel', 'Indigo', 'Ochre', 'lamp', 'buzzer', 'beacon', 'The enclosure had a matte finish.'],
    ['urn', 'Circle', 'Square', 'stripe mark', 'dot mark', 'ring mark', 'The bench was wiped before the session.'],
    ['test card', 'Ivory', 'Pearl', 'corner mark', 'border stripe', 'center dot', 'The tables were arranged in a semicircle.'],
  ],
};

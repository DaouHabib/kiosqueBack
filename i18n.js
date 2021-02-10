import i18n from "i18next";
import { reactI18nextModule } from "react-i18next";
import detector from "i18next-browser-languagedetector";
import backend from "i18next-xhr-backend";

import translationEN from '../languages/en/translation.json';
import translationDE from '../languages/de/translation.json';
import translationFR from '../languages/fr/translation.json';

// the translations
const resources = {
  en: {
    translation: translationEN
  },
  de: {
    translation: translationDE
  },
  fr: {
    translation: translationFR
  }
};

i18n
  .use(detector)
  .use(backend)
  .use(reactI18nextModule) // passes i18n down to react-i18next
  .init({
    resources,
    lng: "en",

    keySeparator: false, // we do not use keys in form messages.welcome

    interpolation: {
      escapeValue: false // react already safes from xss
    },
    react: {
      wait: true,
      bindI18n: 'languageChanged loaded',
    },
 //   debug: true,
    
  });

export default i18n;

#---Imports----
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
#---Imports----
random_seed = 42
np.random.seed(random_seed)



# ---- tain,test,split ----
def t_t_s(working_df):
    feature_cols = [
        'civilian_targeting','fatalities','violence_against_women',
        'sub_event_type_Abduction/forced disappearance',
        'sub_event_type_Air/drone strike',
        'sub_event_type_Armed clash',
        'sub_event_type_Attack',
        'sub_event_type_Government regains territory',
        'sub_event_type_Grenade',
        'sub_event_type_Non-state actor overtakes territory',
        'sub_event_type_Remote explosive/landmine/IED',
        'sub_event_type_Sexual violence',
        'sub_event_type_Shelling/artillery/missile attack',
        'sub_event_type_Suicide bomb',
        'notes_emb_0', 'notes_emb_1', 'notes_emb_2', 'notes_emb_3', 'notes_emb_4', 'notes_emb_5', 'notes_emb_6',
        'notes_emb_7', 'notes_emb_8', 'notes_emb_9', 'notes_emb_10', 'notes_emb_11', 'notes_emb_12', 'notes_emb_13',
        'notes_emb_14', 'notes_emb_15', 'notes_emb_16', 'notes_emb_17', 'notes_emb_18', 'notes_emb_19', 'notes_emb_20',
        'notes_emb_21', 'notes_emb_22', 'notes_emb_23', 'notes_emb_24', 'notes_emb_25', 'notes_emb_26', 'notes_emb_27',
        'notes_emb_28', 'notes_emb_29', 'notes_emb_30', 'notes_emb_31', 'notes_emb_32', 'notes_emb_33', 'notes_emb_34',
        'notes_emb_35', 'notes_emb_36', 'notes_emb_37', 'notes_emb_38', 'notes_emb_39', 'notes_emb_40', 'notes_emb_41',
        'notes_emb_42', 'notes_emb_43', 'notes_emb_44', 'notes_emb_45', 'notes_emb_46', 'notes_emb_47', 'notes_emb_48',
        'notes_emb_49', 'notes_emb_50', 'notes_emb_51', 'notes_emb_52', 'notes_emb_53', 'notes_emb_54', 'notes_emb_55',
        'notes_emb_56', 'notes_emb_57', 'notes_emb_58', 'notes_emb_59', 'notes_emb_60', 'notes_emb_61', 'notes_emb_62',
        'notes_emb_63', 'notes_emb_64', 'notes_emb_65', 'notes_emb_66', 'notes_emb_67', 'notes_emb_68', 'notes_emb_69',
        'notes_emb_70', 'notes_emb_71', 'notes_emb_72', 'notes_emb_73', 'notes_emb_74', 'notes_emb_75', 'notes_emb_76',
        'notes_emb_77', 'notes_emb_78', 'notes_emb_79', 'notes_emb_80', 'notes_emb_81', 'notes_emb_82', 'notes_emb_83',
        'notes_emb_84', 'notes_emb_85', 'notes_emb_86', 'notes_emb_87', 'notes_emb_88', 'notes_emb_89', 'notes_emb_90',
        'notes_emb_91', 'notes_emb_92', 'notes_emb_93', 'notes_emb_94', 'notes_emb_95', 'notes_emb_96', 'notes_emb_97',
        'notes_emb_98', 'notes_emb_99', 'notes_emb_100', 'notes_emb_101', 'notes_emb_102', 'notes_emb_103',
        'notes_emb_104', 'notes_emb_105', 'notes_emb_106', 'notes_emb_107', 'notes_emb_108', 'notes_emb_109',
        'notes_emb_110', 'notes_emb_111', 'notes_emb_112', 'notes_emb_113', 'notes_emb_114', 'notes_emb_115',
        'notes_emb_116', 'notes_emb_117', 'notes_emb_118', 'notes_emb_119', 'notes_emb_120', 'notes_emb_121',
        'notes_emb_122', 'notes_emb_123', 'notes_emb_124', 'notes_emb_125', 'notes_emb_126', 'notes_emb_127',
        'notes_emb_128', 'notes_emb_129', 'notes_emb_130', 'notes_emb_131', 'notes_emb_132', 'notes_emb_133',
        'notes_emb_134', 'notes_emb_135', 'notes_emb_136', 'notes_emb_137', 'notes_emb_138', 'notes_emb_139',
        'notes_emb_140', 'notes_emb_141', 'notes_emb_142', 'notes_emb_143', 'notes_emb_144', 'notes_emb_145',
        'notes_emb_146', 'notes_emb_147', 'notes_emb_148', 'notes_emb_149', 'notes_emb_150', 'notes_emb_151',
        'notes_emb_152', 'notes_emb_153', 'notes_emb_154', 'notes_emb_155', 'notes_emb_156', 'notes_emb_157',
        'notes_emb_158', 'notes_emb_159', 'notes_emb_160', 'notes_emb_161', 'notes_emb_162', 'notes_emb_163',
        'notes_emb_164', 'notes_emb_165', 'notes_emb_166', 'notes_emb_167', 'notes_emb_168', 'notes_emb_169',
        'notes_emb_170', 'notes_emb_171', 'notes_emb_172', 'notes_emb_173', 'notes_emb_174', 'notes_emb_175',
        'notes_emb_176', 'notes_emb_177', 'notes_emb_178', 'notes_emb_179', 'notes_emb_180', 'notes_emb_181',
        'notes_emb_182', 'notes_emb_183', 'notes_emb_184', 'notes_emb_185', 'notes_emb_186', 'notes_emb_187',
        'notes_emb_188', 'notes_emb_189', 'notes_emb_190', 'notes_emb_191', 'notes_emb_192', 'notes_emb_193',
        'notes_emb_194', 'notes_emb_195', 'notes_emb_196', 'notes_emb_197', 'notes_emb_198', 'notes_emb_199',
        'notes_emb_200', 'notes_emb_201', 'notes_emb_202', 'notes_emb_203', 'notes_emb_204', 'notes_emb_205',
        'notes_emb_206', 'notes_emb_207', 'notes_emb_208', 'notes_emb_209', 'notes_emb_210', 'notes_emb_211',
        'notes_emb_212', 'notes_emb_213', 'notes_emb_214', 'notes_emb_215', 'notes_emb_216', 'notes_emb_217',
        'notes_emb_218', 'notes_emb_219', 'notes_emb_220', 'notes_emb_221', 'notes_emb_222', 'notes_emb_223',
        'notes_emb_224', 'notes_emb_225', 'notes_emb_226', 'notes_emb_227', 'notes_emb_228', 'notes_emb_229',
        'notes_emb_230', 'notes_emb_231', 'notes_emb_232', 'notes_emb_233', 'notes_emb_234', 'notes_emb_235',
        'notes_emb_236', 'notes_emb_237', 'notes_emb_238', 'notes_emb_239', 'notes_emb_240', 'notes_emb_241',
        'notes_emb_242', 'notes_emb_243', 'notes_emb_244', 'notes_emb_245', 'notes_emb_246', 'notes_emb_247',
        'notes_emb_248', 'notes_emb_249', 'notes_emb_250', 'notes_emb_251', 'notes_emb_252', 'notes_emb_253',
        'notes_emb_254', 'notes_emb_255', 'notes_emb_256', 'notes_emb_257', 'notes_emb_258', 'notes_emb_259',
        'notes_emb_260', 'notes_emb_261', 'notes_emb_262', 'notes_emb_263', 'notes_emb_264', 'notes_emb_265',
        'notes_emb_266', 'notes_emb_267', 'notes_emb_268', 'notes_emb_269', 'notes_emb_270', 'notes_emb_271',
        'notes_emb_272', 'notes_emb_273', 'notes_emb_274', 'notes_emb_275', 'notes_emb_276', 'notes_emb_277',
        'notes_emb_278', 'notes_emb_279', 'notes_emb_280', 'notes_emb_281', 'notes_emb_282', 'notes_emb_283',
        'notes_emb_284', 'notes_emb_285', 'notes_emb_286', 'notes_emb_287', 'notes_emb_288', 'notes_emb_289',
        'notes_emb_290', 'notes_emb_291', 'notes_emb_292', 'notes_emb_293', 'notes_emb_294', 'notes_emb_295',
        'notes_emb_296', 'notes_emb_297', 'notes_emb_298', 'notes_emb_299', 'notes_emb_300', 'notes_emb_301',
        'notes_emb_302', 'notes_emb_303', 'notes_emb_304', 'notes_emb_305', 'notes_emb_306', 'notes_emb_307',
        'notes_emb_308', 'notes_emb_309', 'notes_emb_310', 'notes_emb_311', 'notes_emb_312', 'notes_emb_313',
        'notes_emb_314', 'notes_emb_315', 'notes_emb_316', 'notes_emb_317', 'notes_emb_318', 'notes_emb_319',
        'notes_emb_320', 'notes_emb_321', 'notes_emb_322', 'notes_emb_323', 'notes_emb_324', 'notes_emb_325',
        'notes_emb_326', 'notes_emb_327', 'notes_emb_328', 'notes_emb_329', 'notes_emb_330', 'notes_emb_331',
        'notes_emb_332', 'notes_emb_333', 'notes_emb_334', 'notes_emb_335', 'notes_emb_336', 'notes_emb_337',
        'notes_emb_338', 'notes_emb_339', 'notes_emb_340', 'notes_emb_341', 'notes_emb_342', 'notes_emb_343',
        'notes_emb_344', 'notes_emb_345', 'notes_emb_346', 'notes_emb_347', 'notes_emb_348', 'notes_emb_349',
        'notes_emb_350', 'notes_emb_351', 'notes_emb_352', 'notes_emb_353', 'notes_emb_354', 'notes_emb_355',
        'notes_emb_356', 'notes_emb_357', 'notes_emb_358', 'notes_emb_359', 'notes_emb_360', 'notes_emb_361',
        'notes_emb_362', 'notes_emb_363', 'notes_emb_364', 'notes_emb_365', 'notes_emb_366', 'notes_emb_367',
        'notes_emb_368', 'notes_emb_369', 'notes_emb_370', 'notes_emb_371', 'notes_emb_372', 'notes_emb_373',
        'notes_emb_374', 'notes_emb_375', 'notes_emb_376', 'notes_emb_377', 'notes_emb_378', 'notes_emb_379',
        'notes_emb_380', 'notes_emb_381', 'notes_emb_382', 'notes_emb_383'
    ]
    X = working_df[feature_cols]
    y = working_df['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.4,
        random_state=random_seed,
        stratify=y
    )
    return X_train, X_test, y_train, y_test

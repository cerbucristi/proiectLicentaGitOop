from dataset_loader import MegaVulDataset
from logistic_regression_model import LogisticRegressionModel
from codebert_model import CodeBertModel
from code_processor import CodePreprocessor

dataset = MegaVulDataset()
samples = dataset.load(limit=500)
codeProcessor = CodePreprocessor()

model = LogisticRegressionModel()
model.load("artifacts/classical/model.pkl")
# model.train(samples)
# model.save("artifacts/classical/model.pkl")

codebert = CodeBertModel()
codebert.load("artifacts/codebert/model")

codeString = """
static int rndis_query_response(USBNetState *s, rndis_query_msg_type *buf, unsigned int length) { rndis_query_cmplt_type *resp; /* oid_supported_list is the largest data reply */ uint8_t infobuf[sizeof(oid_supported_list)]; uint32_t bufoffs, buflen; int infobuflen; unsigned int resplen; bufoffs = le32_to_cpu(buf->InformationBufferOffset) + 8; buflen = le32_to_cpu(buf->InformationBufferLength); if (buflen > length || bufoffs >= length || bufoffs + buflen > length) { return USB_RET_STALL; } infobuflen = ndis_query(s, le32_to_cpu(buf->OID), bufoffs + (uint8_t *) buf, buflen, infobuf, sizeof(infobuf)); resplen = sizeof(rndis_query_cmplt_type) + ((infobuflen < 0) ? 0 : infobuflen); resp = rndis_queue_response(s, resplen); if (!resp) return USB_RET_STALL; resp->MessageType = cpu_to_le32(RNDIS_QUERY_CMPLT); resp->RequestID = buf->RequestID; /* Still LE in msg buffer */ resp->MessageLength = cpu_to_le32(resplen); if (infobuflen < 0) { /* OID not supported */ resp->Status = cpu_to_le32(RNDIS_STATUS_NOT_SUPPORTED); resp->InformationBufferLength = cpu_to_le32(0); resp->InformationBufferOffset = cpu_to_le32(0); return 0; } resp->Status = cpu_to_le32(RNDIS_STATUS_SUCCESS); resp->InformationBufferOffset = cpu_to_le32(infobuflen ? sizeof(rndis_query_cmplt_type) - 8 : 0); resp->InformationBufferLength = cpu_to_le32(infobuflen); memcpy(resp + 1, infobuf, infobuflen); return 0; }
"""
print("Logistic regression prediction")
print(model.predict(codeProcessor.clean(codeString)))

# codebert.train(samples)
# codebert.save("artifacts/codebert/model")
print("Code Bert prediction:")
print(codebert.predict(codeProcessor.clean(codeString)))



# model = CodeBertModel()
# model.train(samples)      #  samples folosite
# prob = model.predict(code)
# model.save("artifacts/codebert")